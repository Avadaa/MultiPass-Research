"""DDGS metasearch HTTP client via the localhost `ddgs api` sidecar.

The sidecar is auto-started when this module is imported (blocking, so
search() never proceeds before it's ready). If a sidecar is already
listening on 127.0.0.1:8000 we reuse it instead of spawning a duplicate.
The autostart only runs when `config.SEARCH_ENGINE == "ddgs"`, so brave-
only runs don't pay for a sidecar they won't use.

Why a sidecar:  the HTTP call from this script to 127.0.0.1 is loopback
traffic and bypasses the VPN entirely. Whether ddgs's own outbound search
requests go through the VPN depends on the sidecar process's own split-
tunnel status - a single clean decision point instead of inheriting
python.exe's routing.

Exposes:
    search(query, count, backend, region, endpoint)
        -> list[{"title", "url", "description"}]

Non-200 responses (including 429 rate limits) raise immediately with a
full traceback - rate limits must be unmissable. Connection failures
(sidecar not running) exit cleanly with a hint.

Standalone CLI:  python search_ddg.py "query"
Browse live schema at http://127.0.0.1:8000/docs.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from config import DDGS_BACKEND, LOCALE, SEARCH_ENGINE, SEARCH_RESULT_COUNT


DEFAULT_ENDPOINT = "http://127.0.0.1:8000"
DEFAULT_BACKEND = DDGS_BACKEND


# ---------------------------------------------------------------------------
# Sidecar autostart
# ---------------------------------------------------------------------------

_DDGS_HOST = "127.0.0.1"
_DDGS_PORT = 8000
_READY_TIMEOUT = 30.0
_READY_POLL = 0.5
_proc: subprocess.Popen | None = None
_job_handle: int | None = None  # Windows Job Object handle (kill-on-close)


# ---------------------------------------------------------------------------
# OS-specific orphan-prevention hardening
#
# atexit covers graceful exits, exceptions, and Ctrl+C. It does NOT cover
# SIGKILL, segfaults, OOM kills, Task Manager termination, or os._exit().
# To stop the sidecar from outliving a hard parent crash:
#   - Linux: child calls prctl(PR_SET_PDEATHSIG, SIGTERM) before exec; kernel
#            delivers SIGTERM to the child as soon as the parent dies.
#   - Windows: parent creates a Job Object with KILL_ON_JOB_CLOSE and assigns
#            the child to it; when the parent dies (any reason), the OS closes
#            the job handle and kills every process in the job.
#   - macOS: no PR_SET_PDEATHSIG equivalent; falls through unhardened. Next-run
#            port-reuse handles cleanup-after-crash in practice.
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JobObjectExtendedLimitInformation = 9

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_void_p),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    def _create_kill_on_close_job() -> int:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())

        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job,
            _JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            err = ctypes.get_last_error()
            kernel32.CloseHandle(job)
            raise ctypes.WinError(err)
        return job

    def _assign_to_job(job: int, proc: subprocess.Popen) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        if not kernel32.AssignProcessToJobObject(job, int(proc._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

elif sys.platform == "linux":
    import ctypes

    _PR_SET_PDEATHSIG = 1

    def _set_pdeathsig_in_child() -> None:
        """preexec_fn: in the child, ask the kernel to SIGTERM us when our
        parent dies. Best-effort — silently no-ops if libc/prctl unavailable."""
        try:
            libc = ctypes.CDLL(None)
            libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
        except (OSError, AttributeError):
            pass


def _port_listening(host: str, port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def _wait_ready(endpoint: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    url = f"{endpoint.rstrip('/')}/docs"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(_READY_POLL)
    return False


def _stop_sidecar() -> None:
    """Cross-platform graceful stop of the spawned sidecar.

    - Windows: send CTRL_BREAK_EVENT to the new process group.
    - POSIX:   SIGTERM the entire session (started via start_new_session=True),
               so uvicorn workers and any grandchildren die with their parent.

    Falls back to proc.terminate() / proc.kill() if the group-level signal
    can't be delivered (e.g. process already gone)."""
    global _proc
    p = _proc
    if p is None or p.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            os.kill(p.pid, signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except (OSError, ValueError, ProcessLookupError):
        p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if sys.platform != "win32":
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        p.kill()


def _ensure_sidecar() -> None:
    """Block until the ddgs sidecar at 127.0.0.1:8000 is responding.

    Idempotent: returns immediately if we already spawned a live sidecar,
    or if a sidecar (someone else's) is already answering at /docs. Raises
    on failure (ddgs not on PATH, port held by something incompatible,
    sidecar didn't come up within _READY_TIMEOUT).

    Cross-platform spawn:
      - Windows: CREATE_NEW_PROCESS_GROUP so we can send CTRL_BREAK_EVENT.
      - POSIX:   start_new_session=True so we can SIGTERM the whole group.
    """
    global _proc
    if _proc is not None and _proc.poll() is None:
        return  # already spawned and alive — nothing to do.

    if _port_listening(_DDGS_HOST, _DDGS_PORT):
        if _wait_ready(DEFAULT_ENDPOINT, 5.0):
            print(f"[ddgs] reusing existing sidecar at {DEFAULT_ENDPOINT}")
            return
        raise RuntimeError(
            f"port {_DDGS_PORT} is in use but {DEFAULT_ENDPOINT}/docs did "
            f"not respond - free the port or stop whatever is on it"
        )

    ddgs_bin = shutil.which("ddgs")
    if not ddgs_bin:
        raise RuntimeError(
            "`ddgs` not on PATH. Install with: pip install ddgs[api]"
        )

    global _job_handle
    print(f"[ddgs] spawning sidecar: {ddgs_bin} api")
    popen_kwargs: dict = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
        if sys.platform == "linux":
            popen_kwargs["preexec_fn"] = _set_pdeathsig_in_child

    _proc = subprocess.Popen([ddgs_bin, "api"], **popen_kwargs)
    atexit.register(_stop_sidecar)

    # Windows: assign the child to a kill-on-close Job Object so it dies with
    # us even on TerminateProcess / segfault / Task Manager kill.
    if sys.platform == "win32":
        try:
            _job_handle = _create_kill_on_close_job()
            _assign_to_job(_job_handle, _proc)
        except OSError as e:
            print(f"[ddgs] warning: Job Object hardening unavailable ({e!r}); "
                  f"sidecar may orphan if parent crashes hard")
            _job_handle = None

    if not _wait_ready(DEFAULT_ENDPOINT, _READY_TIMEOUT):
        _stop_sidecar()
        raise RuntimeError(
            f"ddgs sidecar at {DEFAULT_ENDPOINT} did not come up within "
            f"{_READY_TIMEOUT:.0f}s"
        )
    print(f"[ddgs] sidecar ready at {DEFAULT_ENDPOINT} (pid={_proc.pid})")


if SEARCH_ENGINE == "ddgs":
    _ensure_sidecar()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    query: str,
    count: int = SEARCH_RESULT_COUNT,
    backend: str = DEFAULT_BACKEND,
    region: str = LOCALE,
    endpoint: str = DEFAULT_ENDPOINT,
) -> list[dict]:
    """Normalized to [{"title", "url", "description"}, ...]."""
    params = urllib.parse.urlencode({
        "query": query,
        "max_results": count,
        "backend": backend,
        "region": region,
    })
    url = f"{endpoint.rstrip('/')}/search/text?{params}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
        # urlopen auto-raises HTTPError for 4xx/5xx; this catches 2xx-non-200
        # (e.g. 202 when upstream DuckDuckGo throttles the scraper). Re-raise
        # as HTTPError so the orchestration layer can treat every non-200
        # uniformly.
        if r.status != 200:
            raise urllib.error.HTTPError(
                req.full_url, r.status,
                f"DDGS returned HTTP {r.status} (expected 200): "
                f"{body[:500].decode('utf-8', errors='replace')!r}",
                r.headers, None,
            )
        raw = json.loads(body)
    # Server wraps the list in {"results": [...]}; older/other endpoints may
    # return a flat list. Accept both.
    if isinstance(raw, dict):
        for key in ("results", "items", "data"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise RuntimeError(f"ddgs returned unexpected shape: {raw!r}")
    return [
        {
            "title": r.get("title") or "",
            "url": r.get("href") or r.get("url") or "",
            "description": r.get("body") or r.get("description") or "",
        }
        for r in raw
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default=DEFAULT_BACKEND,
                        help="ddgs backend(s). e.g. 'google,duckduckgo' or 'auto'")
    parser.add_argument("--count", type=int, default=SEARCH_RESULT_COUNT)
    parser.add_argument("--region", default=LOCALE)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                        help=f"ddgs server base URL (default {DEFAULT_ENDPOINT})")
    parser.add_argument("query", nargs="*")
    args = parser.parse_args()

    query = " ".join(args.query) or "how to detect AI text in cover letters"
    print(f"[query]    {query}")
    print(f"[backend]  {args.backend}")
    print(f"[region]   {args.region}")
    print(f"[endpoint] {args.endpoint}\n")

    # Standalone CLI: autostart the sidecar even if config.SEARCH_ENGINE != 'ddgs'.
    # (The import-time autostart is gated on that to avoid spawning during
    # brave-only pipeline runs; here we know the user wants ddgs.)
    if args.endpoint == DEFAULT_ENDPOINT:
        _ensure_sidecar()

    try:
        results = search(
            query,
            count=args.count,
            backend=args.backend,
            region=args.region,
            endpoint=args.endpoint,
        )
    except urllib.error.HTTPError:
        raise
    except urllib.error.URLError as e:
        sys.exit(f"request failed: {e!r}. Is `ddgs api` running at {args.endpoint}?")

    if not results:
        print("(no results)")
        return
    print(f"({len(results)} result(s))\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title'] or '(no title)'}")
        print(f"   {r['url']}")
        if r["description"]:
            snippet = r["description"].strip().replace("\n", " ")
            print(f"   {snippet[:200]}")
        print()


if __name__ == "__main__":
    main()

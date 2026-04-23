"""Daddy decides. Minions execute.
  - spawn_minions(tasks): delegate N subtasks in parallel; each task has a tool choice
  - talk_to_user(message): end the loop by sending a final message to the user

Minion tools:
  - free_think:   answer from own knowledge
  - pull_website: fetch a URL and answer from its content
  - elaborate:    re-examine an already-fetched URL (from daddy's cache) with a
                  new, deeper prompt. Content is served from cache, no re-fetch.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import sys
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# crawl4ai prints banner chars (→, ✓) and leaks subprocess fds on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
warnings.filterwarnings("ignore", category=ResourceWarning)

from crawl4ai import AsyncWebCrawler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI


SERVER_URL = "http://localhost:8080"
CONTEXT_LIMIT = 131072  # per-slot (-c 262144 with --parallel 2)
LOG_DIR = Path(__file__).parent / "logs"


class _Tee:
    """Mirror writes to the real stream AND a log file. Survives encoding errors."""
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file
    def write(self, data: str) -> int:
        try:
            n = self._stream.write(data)
        except Exception:
            n = len(data)
        try:
            self._log.write(data)
            self._log.flush()
        except Exception:
            pass
        return n
    def flush(self) -> None:
        for s in (self._stream, self._log):
            try: s.flush()
            except Exception: pass
    def __getattr__(self, name):
        return getattr(self._stream, name)


def install_run_logger() -> Path:
    """Tee stdout+stderr into ./logs/run_<timestamp>.log. Returns the log path."""
    LOG_DIR.mkdir(exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = LOG_DIR / f"run_{ts}.log"
    f = open(path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)
    return path


def count_tokens(text: str) -> int:
    """Ask llama-server to tokenize the text. Falls back to char/4 estimate."""
    try:
        req = urllib.request.Request(
            f"{SERVER_URL}/tokenize",
            data=json.dumps({"content": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return len(json.loads(r.read()).get("tokens", []))
    except Exception:
        return len(text) // 4


class ChatLlamaServer(ChatOpenAI):
    """ChatOpenAI that preserves llama.cpp / DeepSeek-style `reasoning_content`."""

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump()
        for gen, choice in zip(result.generations, raw.get("choices", [])):
            reasoning = (choice.get("message") or {}).get("reasoning_content")
            if reasoning and isinstance(gen.message, AIMessage):
                gen.message.additional_kwargs["reasoning_content"] = reasoning
        return result


FETCH_CHAR_LIMIT = 20000


def fetch_website(url: str) -> str:
    """Fetch URL, return markdown. Raises on failure."""
    async def _run():
        async with AsyncWebCrawler() as c:
            result = await c.arun(url=url)
        if not result.success:
            raise RuntimeError(result.error_message or "fetch failed")
        md = result.markdown or ""
        return md if isinstance(md, str) else getattr(md, "raw_markdown", "")
    return asyncio.run(_run())


@dataclass
class MinionTask:
    prompt: str
    tool: str                      # "free_think" | "pull_website" | "elaborate"
    url: str = ""                  # used when tool is pull_website or elaborate
    fetched_content: str | None = None  # pre-populated by daddy for elaborate


@dataclass
class MinionResult:
    minion_name: str
    task: MinionTask
    answers_question: bool
    question_if_yes: str
    summary: str
    fetched_content: str | None = None  # returned by pull_website for daddy's cache


MINION_SCHEMA = {
    "type": "object",
    "properties": {
        "answers_question": {"type": "boolean"},
        "question_if_yes": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["answers_question", "question_if_yes", "summary"],
    "additionalProperties": False,
}


MINION_OUTPUT_SPEC = (
    'Respond as JSON:\n'
    '{\n'
    '  "answers_question": <true if you substantively answer the prompt; '
    'false otherwise>,\n'
    '  "question_if_yes": "<the exact question you are answering, or \\"\\" '
    'if answers_question is false>",\n'
    '  "summary": "<answer, or \\"task not possible\\" if '
    'answers_question is false>"\n'
    '}'
)

NAV_CLARIFICATION = (
    "IMPORTANT: If the page merely MENTIONS the topic (e.g., as a link, "
    "heading, nav item, or brief reference) WITHOUT actually providing the "
    "information the user asks for, set answers_question=false. Merely "
    "containing the words of the question is NOT answering it. The page "
    "must contain the substantive information that would answer the "
    "question."
)


class Minion:
    def __init__(self, name: str, llm: ChatLlamaServer):
        self.name = name
        self.llm = llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "minion_answer",
                    "strict": True,
                    "schema": MINION_SCHEMA,
                },
            },
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)

    def _system_prompt(self, task: MinionTask, fetched: str | None) -> str:
        if task.tool == "pull_website":
            if fetched is None:
                return (
                    "You were given the tool pull_website but the fetch failed. "
                    'Return answers_question=false, question_if_yes="", '
                    'summary="task not possible".\n\n' + MINION_OUTPUT_SPEC
                )
            return (
                "You are a minion. You have ONE tool: pull_website.\n"
                f"It has already been called on: {task.url}\n"
                "The page's markdown content is below. Use ONLY this content "
                "to answer the user's question.\n\n"
                f"{NAV_CLARIFICATION}\n\n"
                "--- BEGIN FETCHED CONTENT ---\n"
                f"{fetched[:FETCH_CHAR_LIMIT]}\n"
                "--- END FETCHED CONTENT ---\n\n"
                + MINION_OUTPUT_SPEC
            )
        if task.tool == "elaborate":
            if fetched is None:
                return (
                    "You were asked to elaborate on a page but no content "
                    'was available. Return answers_question=false, '
                    'question_if_yes="", summary="task not possible '
                    '(no cached content)".\n\n' + MINION_OUTPUT_SPEC
                )
            return (
                "You are an ELABORATION minion. A page has already been "
                "scanned and deemed relevant by daddy. Your job is to extract "
                "COMPREHENSIVE detail from this page that answers the user's "
                "prompt. Be thorough, not terse: quote directly when useful, "
                "list every relevant item, include specific names, numbers, "
                "and verbatim phrases. Daddy has specifically asked for depth "
                "on this page — give it to him.\n"
                f"Page URL: {task.url}\n\n"
                "--- BEGIN FETCHED CONTENT ---\n"
                f"{fetched[:FETCH_CHAR_LIMIT]}\n"
                "--- END FETCHED CONTENT ---\n\n"
                + MINION_OUTPUT_SPEC
            )
        return (
            "You are a minion with no tools. Answer the user's question from "
            "your own knowledge. If you cannot answer or lack information, "
            "set answers_question=false and summary=\"task not possible\".\n\n"
            + MINION_OUTPUT_SPEC
        )

    def run(self, task: MinionTask) -> MinionResult:
        fetched: str | None = task.fetched_content

        if task.tool == "pull_website":
            try:
                fetched = fetch_website(task.url)
            except Exception as e:
                print(f"  [{self.name}] fetch failed for {task.url}: {e}")
                return MinionResult(
                    self.name, task,
                    answers_question=False,
                    question_if_yes="",
                    summary="task not possible (fetch failed)",
                )

        resp = self.llm.invoke([
            SystemMessage(content=self._system_prompt(task, fetched)),
            HumanMessage(content=task.prompt),
        ])
        parsed = json.loads(resp.content)
        return MinionResult(
            self.name, task,
            answers_question=parsed["answers_question"],
            question_if_yes=parsed["question_if_yes"],
            summary=parsed["summary"],
            fetched_content=fetched if task.tool == "pull_website" else None,
        )


DADDY_SYSTEM = """You are Daddy, an orchestrator with NO external capabilities
of your own. To get anything from the outside world, you delegate to minions.

HARD RULE — ELABORATE BEFORE YOU ANSWER:
For EVERY URL that a pull_website scan returned answers_question=true, you
MUST call elaborate on that URL before you call talk_to_user. No exceptions.
A scan gives you a verdict; an elaborate gives you the answer. If you skip
elaborate and go straight to talk_to_user, your reply will be thin and the
user will have to ask again. The elaborate-call lets you reframe the prompt
now that you know what the page is about, and it is FREE (content is served
from cache, no re-fetch).

Sequence: scan (pull_website) -> elaborate on every positive hit -> talk_to_user.

YOUR JOB AS AN ENGINEER:
Your ability to delegate to minions exists to save YOUR context for synthesis,
not to shift effort. Engineer each minion's prompt with care:
  - EXPAND the user's query. Do not just forward it. Minions get one shot;
    give them enough context and explicit instructions to answer fully on the
    first try.
  - Ask for COMPLETE answers with specific details. Demand direct quotes,
    numbers, names, or explicit "not found" statements when relevant.
  - Work backward from what YOU need to know to answer the user, to the
    minimal set of minion calls that gets you there.
You are the engineer. You are the daddy.

At each turn you choose EXACTLY ONE action:

  spawn_minions — delegate one or more tasks in PARALLEL. Each task has:
      prompt: the instruction or question for that minion
      tool:   "free_think" | "pull_website" | "elaborate"
      url:    the URL if tool is pull_website or elaborate; otherwise ""

  Available minion tools:
    - free_think:   the minion answers from its own knowledge. Use for
                    reasoning, general knowledge, or transforming data.
    - pull_website: the minion fetches the URL and scans it briefly for
                    relevance. Use for INITIAL scanning of unknown URLs.
                    Output is a short relevance verdict + brief summary.
                    ONE minion scans ONE URL — for N URLs, spawn N minions.
    - elaborate:    the minion revisits an already-scanned URL (its content
                    is served from cache — no re-fetch) with a deeper,
                    more specific prompt. Use when a pull_website scan
                    looked promising and you want comprehensive detail,
                    verbatim quotes, or deeper extraction before replying
                    to the user. You must pass the SAME url that a previous
                    pull_website minion already scanned.

  Typical workflow:
    Turn 1: spawn_minions with pull_website across candidate URLs (scan).
    Turn 2: spawn_minions with elaborate on the promising URL(s) (deepen).
    Turn 3: talk_to_user with the synthesized answer.

  Minions perform best with ONE atomic subtask each. Split compound work
  into separate tasks. Independent tasks run concurrently, so prefer
  parallel fan-out over sequential spawning.

  If a minion reports answers_question=false, that minion's task could not be
  fulfilled. Factor that into your final reply.

  talk_to_user — end the conversation with a final message to the user.

Respond ONLY with JSON (always include all three top-level fields; unused
ones are empty):
  {"action": "spawn_minions",
   "tasks": [{"prompt": "...", "tool": "pull_website", "url": "https://..."}],
   "message": ""}
  {"action": "talk_to_user", "tasks": [], "message": "<final reply>"}
"""

DADDY_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["spawn_minions", "talk_to_user"]},
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "tool": {
                        "type": "string",
                        "enum": ["free_think", "pull_website", "elaborate"],
                    },
                    "url": {"type": "string"},
                },
                "required": ["prompt", "tool", "url"],
                "additionalProperties": False,
            },
        },
        "message": {"type": "string"},
    },
    "required": ["action", "tasks", "message"],
    "additionalProperties": False,
}


class Daddy:
    MAX_TURNS = 10

    def __init__(self, llm: ChatLlamaServer):
        self.minion_llm = llm
        self.daddy_llm = llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "daddy_action",
                    "strict": True,
                    "schema": DADDY_SCHEMA,
                },
            }
        ).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
        self._spawned = 0
        self._page_cache: dict[str, str] = {}

    def spawn(self) -> Minion:
        self._spawned += 1
        return Minion(f"minion-{self._spawned}", self.minion_llm)

    def _prepare_elaborate(self, task: MinionTask) -> None:
        """Fill task.fetched_content from cache; fall back to fresh fetch."""
        if task.url in self._page_cache:
            task.fetched_content = self._page_cache[task.url]
            print(f"  [cache hit] {task.url} ({len(task.fetched_content)} chars)")
            return
        print(f"  [cache miss] {task.url} -> fetching fresh")
        try:
            content = fetch_website(task.url)
            self._page_cache[task.url] = content
            task.fetched_content = content
        except Exception as e:
            print(f"  [fetch failed during elaborate] {task.url}: {e}")
            task.fetched_content = None

    def _run_minions(self, tasks: list[MinionTask]) -> list[MinionResult]:
        for t in tasks:
            if t.tool == "elaborate":
                self._prepare_elaborate(t)

        minions = [self.spawn() for _ in tasks]
        print(f"[daddy spawns {len(minions)} minion(s) in parallel]")
        for m, t in zip(minions, tasks):
            if t.tool in ("pull_website", "elaborate"):
                tool_desc = f"{t.tool}({t.url})"
            else:
                tool_desc = t.tool
            print(f"  [daddy -> {m.name} | {tool_desc}] {t.prompt}")

        with ThreadPoolExecutor(max_workers=len(minions)) as pool:
            results = list(pool.map(
                lambda pair: pair[0].run(pair[1]),
                zip(minions, tasks),
            ))

        for r in results:
            if r.fetched_content is not None:
                self._page_cache[r.task.url] = r.fetched_content

        for r in results:
            status = "YES" if r.answers_question else "NO "
            print(f"  [{r.minion_name} -> daddy | {status}] {r.summary}")
            if r.answers_question and r.question_if_yes:
                print(f"       (answered: {r.question_if_yes!r})")
        print()
        return results

    def _print_context(self, history: list, label: str) -> None:
        text = "\n".join(
            msg.content if isinstance(msg.content, str) else str(msg.content)
            for msg in history
        )
        n_toks = count_tokens(text)
        pct = n_toks / CONTEXT_LIMIT * 100
        print(
            f"  [daddy context | {label}: {len(history)} msgs / "
            f"~{n_toks:,} tok / {pct:.1f}% of {CONTEXT_LIMIT:,}]"
        )

    def handle(self, user_task: str) -> str:
        t0 = time.perf_counter()
        history: list = [
            SystemMessage(content=DADDY_SYSTEM),
            HumanMessage(content=user_task),
        ]
        print(f"[user -> daddy] {user_task}\n")
        self._print_context(history, "after user task")

        for turn in range(1, self.MAX_TURNS + 1):
            resp = self.daddy_llm.invoke(history)
            history.append(resp)
            usage = resp.usage_metadata or {}
            finish = (resp.response_metadata or {}).get("finish_reason", "?")
            reasoning = resp.additional_kwargs.get("reasoning_content") or ""
            reasoning_toks = count_tokens(reasoning) if reasoning else 0
            content_toks = usage.get("output_tokens", 0) - reasoning_toks
            print(
                f"  [daddy turn {turn} usage: "
                f"~{reasoning_toks:,} thinking + ~{max(content_toks, 0):,} content "
                f"= {usage.get('output_tokens', 0):,} / 32768, finish_reason={finish}]"
            )
            self._print_context(history, f"after daddy turn {turn}")
            action = json.loads(resp.content)

            if action["action"] == "talk_to_user":
                msg = action["message"]
                suspicions = []
                if finish == "length":
                    suspicions.append("finish_reason='length' (hit max_tokens)")
                if len(msg) < 300:
                    suspicions.append(f"message is only {len(msg)} chars")
                last_line = next(
                    (ln.rstrip() for ln in reversed(msg.splitlines()) if ln.strip()),
                    "",
                )
                if last_line and not last_line.rstrip().endswith(
                    (".", "!", "?", ":", "*", ")", '"', "'")
                ):
                    suspicions.append(f"last line ends mid-thought: ...{last_line[-40:]!r}")
                if suspicions:
                    print("  [!! TRUNCATION WARNING] " + "; ".join(suspicions))
                print(f"[daddy -> user] {msg}")
                elapsed = time.perf_counter() - t0
                print(f"\n[total elapsed] {elapsed:.1f}s (query -> final answer)")
                return msg

            if action["action"] == "spawn_minions":
                raw_tasks = action["tasks"]
                if not raw_tasks:
                    raise ValueError("daddy picked spawn_minions with empty task list")
                tasks = [
                    MinionTask(prompt=t["prompt"], tool=t["tool"], url=t.get("url", ""))
                    for t in raw_tasks
                ]
                results = self._run_minions(tasks)
                report = "\n".join(
                    f"{r.minion_name} (tool={r.task.tool}, url={r.task.url or '-'}, "
                    f"prompt={r.task.prompt!r}) -> answers_question={r.answers_question}, "
                    f"question_if_yes={r.question_if_yes!r}, summary={r.summary!r}"
                    for r in results
                )
                history.append(HumanMessage(content=f"Minion reports:\n{report}"))
                self._print_context(history, f"after minion reports (turn {turn})")
                continue

        raise RuntimeError("daddy hit MAX_TURNS without talking to the user")


def main():
    log_path = install_run_logger()
    print(f"[logging this run to {log_path}]\n")

    llm = ChatLlamaServer(
        base_url="http://localhost:8080/v1",
        api_key="not-needed",
        model="local",
        temperature=0.6,
        max_tokens=32768,
    )

    daddy = Daddy(llm)
    daddy.handle("""
  do any of these websites answer any of these questions:
  'how to detect AI text in cover letters' or 'what is required to be a nurse in Finland'
  https://www.coversentry.com/
  https://www.coversentry.com/info/ai-detection-guide
  https://www.coversentry.com/cv-pohjat/sairaanhoitaja
  """)


if __name__ == "__main__":
    main()

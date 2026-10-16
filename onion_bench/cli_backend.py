"""Run the eval through the Claude Code CLI (`claude -p`), i.e. on a Claude
Pro/Max subscription instead of an API key.

Each sample is a separate `claude -p` process, so no state is shared between
calls. Measures taken to keep it comparable to the API runner:
- `--system-prompt` replaces Claude Code's default agent prompt with ours.
- `--tools ""` removes every built-in tool (incl. WebSearch/WebFetch/Bash) and
  `--disallowedTools mcp__*` any MCP tool: no internet. The only tool left is
  `StructuredOutput`, which is how `--json-schema` is implemented.
- `--safe-mode` keeps CLAUDE.md, memory, skills, plugins and hooks out.
- The child runs with a scrubbed environment in an empty, neutrally named
  temp directory outside the repo (Claude Code tells the model its working
  directory path), so no session/remote context or project name leaks in. ANTHROPIC_API_KEY is never passed,
  so calls cannot silently bill an API key.

What Claude Code still adds (verified by capturing the request): a one-line
SDK preamble in the system prompt, a reminder with the account e-mail, and an
environment block (platform, model name, knowledge cutoff, today's date).
None of it identifies an article's source. Every record is tagged
`backend="claude_cli"`, and each call's init/result event is checked: a call
whose tools, MCP servers, credential source or serving model differ from the
expected setup is recorded as `invalid_setup` instead of as an answer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .prompts import OUTPUT_SCHEMA, PROMPT_VERSION, user_message
from .prompts import system_prompt as _system_prompt
from .runner import EvalConfig, Job, _Writer, error_record

log = logging.getLogger(__name__)

# Variables the child needs: binary lookup, credentials in $HOME, egress proxy and CA bundle.
_KEEP_ENV = (
    "PATH", "HOME", "LANG", "LC_ALL", "TMPDIR",
    "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy",
    "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE",
)


def child_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k in _KEEP_ENV}


def command(job: Job, claude_bin: str = "claude", *, system: str | None = None,
            schema: dict | None = None, user: str | None = None) -> list[str]:
    return [
        claude_bin, "-p", user if user is not None else user_message(job.item, job.variant),
        "--model", job.model.id,
        "--effort", job.model.effort,
        "--system-prompt", system or _system_prompt(job.variant),
        "--tools", "",
        "--disallowedTools", "mcp__*",
        "--safe-mode",
        "--no-session-persistence",
        "--max-turns", "2",
        "--output-format", "stream-json", "--verbose",
        "--json-schema", json.dumps(schema or OUTPUT_SCHEMA),
    ]


def _rate_summary(rate: dict | None) -> dict | None:
    """Plan usage as reported by the CLI. `utilization`/`resetsAt` refer to the
    most-used window (e.g. five_hour or a weekly one)."""
    if not rate:
        return None
    windows = {k: v for k, v in (rate.get("unifiedWindows") or {}).items() if isinstance(v, dict)}
    top = max(windows.items(), key=lambda kv: kv[1].get("utilization") or 0, default=(rate.get("rateLimitType"), {}))
    return {
        "status": rate.get("status"),
        "window": top[0],
        "utilization": top[1].get("utilization"),
        "resetsAt": top[1].get("resetsAt") or rate.get("resetsAt"),
        "windows": {k: v.get("utilization") for k, v in windows.items()},
    }


def parse_stream(job: Job, lines: list[str], allow_api_key: bool = False) -> dict:
    """Turn one `claude -p --output-format stream-json` transcript into a record."""
    init, result, rate = None, None, None
    served: set[str] = set()
    for line in lines:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = e.get("type")
        if t == "system" and e.get("subtype") == "init":
            init = e
        elif t == "assistant":
            if m := (e.get("message") or {}).get("model"):
                served.add(m)
        elif t == "rate_limit_event":
            rate = e.get("rate_limit_info")
        elif t == "result":
            result = e
    base = {
        "model": job.model.id, "item_id": job.item["id"], "sample": job.sample,
        "prompt_version": PROMPT_VERSION, "prompt_variant": job.variant, "effort": job.model.effort, "backend": "claude_cli",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if result is None:
        return base | {"status": "api_error", "verdict": None, "p_fake": None, "error": "no result event"}
    served |= set((result.get("modelUsage") or {}).keys())
    usage = result.get("usage") or {}
    base |= {
        "model_served": sorted(served),
        "stop_reason": result.get("stop_reason"),
        "input_tokens": (usage.get("input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0) + (usage.get("cache_read_input_tokens") or 0),
        "output_tokens": usage.get("output_tokens"),
        "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens"),
        "latency_s": round((result.get("duration_ms") or 0) / 1000, 2),
        "equivalent_api_cost_usd": result.get("total_cost_usd"),
        "rate_limit": _rate_summary(rate),
        "raw_text": result.get("result"),
    }
    problems = []
    if init is not None:
        if init.get("tools") != ["StructuredOutput"]:
            problems.append(f"tools={init.get('tools')}")
        if init.get("mcp_servers"):
            problems.append(f"mcp_servers={init.get('mcp_servers')}")
        if init.get("apiKeySource") not in ("none", None) and not allow_api_key:
            problems.append(f"apiKeySource={init.get('apiKeySource')}")
    else:
        problems.append("no init event")
    if served and served != {job.model.id}:
        problems.append(f"served_by={sorted(served)}")
    if problems:
        return base | {"status": "invalid_setup", "verdict": None, "p_fake": None, "error": "; ".join(problems)}

    so = result.get("structured_output")
    if isinstance(so, dict) and so.get("verdict") in ("real", "fake"):
        p = so.get("p_fake")
        p_fake = min(max(float(p), 0.0), 100.0) / 100.0 if isinstance(p, (int, float)) else None
        return base | {"status": "ok", "verdict": so["verdict"], "p_fake": p_fake}
    if result.get("stop_reason") == "refusal":
        return base | {"status": "refusal", "verdict": None, "p_fake": None}
    if result.get("is_error"):
        status = "rate_limited" if "limit" in str(result.get("result", "")).lower() else "api_error"
        return base | {"status": status, "verdict": None, "p_fake": None, "error": str(result.get("result"))[:500]}
    return base | {"status": "parse_error", "verdict": None, "p_fake": None}


async def run_cli(
    cfg: EvalConfig,
    jobs: list[Job],
    responses: Path,
    *,
    concurrency: int = 4,
    pause_at_utilization: float = 0.9,
    timeout_s: float = 900,
    allow_api_key: bool = False,
    claude_bin: str | None = None,
) -> dict:
    claude_bin = claude_bin or shutil.which("claude")
    if not claude_bin:
        raise RuntimeError("`claude` CLI not found on PATH")
    # Claude Code shows the working directory path (and whether it is a git
    # repo) to the model, so use an empty, neutrally named directory outside
    # the repo: a path like ".../onion-bench/..." would hint at satire.
    workdir = Path(tempfile.mkdtemp(prefix="session-"))
    env = child_env()
    sem = asyncio.Semaphore(concurrency)
    out = _Writer(responses)
    counts: dict[str, int] = {}
    state = {"pause_until": 0.0, "stop": False}

    async def one(job: Job) -> None:
        async with sem:
            if state["stop"]:
                return
            while (wait := state["pause_until"] - time.time()) > 0:
                log.warning("usage window nearly exhausted; sleeping %.0f min", wait / 60)
                await asyncio.sleep(min(wait, 300))
            proc = await asyncio.create_subprocess_exec(
                *command(job, claude_bin), cwd=workdir, env=env,
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout_s)
                rec = parse_stream(job, stdout.decode("utf-8", "replace").splitlines(), allow_api_key)
                if rec["status"] == "api_error" and stderr:
                    rec["error"] = (rec.get("error", "") + " | " + stderr.decode("utf-8", "replace")[-400:]).strip(" |")
            except asyncio.TimeoutError:
                proc.kill()
                rec = error_record(job, "api_error", f"timeout after {timeout_s}s") | {"backend": "claude_cli"}
            rl = rec.get("rate_limit") or {}
            if (rl.get("utilization") or 0) >= pause_at_utilization or rec["status"] == "rate_limited":
                state["pause_until"] = max(state["pause_until"], float(rl.get("resetsAt") or time.time() + 1800))
            if rec["status"] == "invalid_setup":
                log.error("invalid setup, stopping run: %s", rec["error"])
                state["stop"] = True
            if rec["status"] == "rate_limited":
                rec["status"] = "api_error"  # retried on the next run
            out.write(rec)
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            n = sum(counts.values())
            if n % 20 == 0 or n == len(jobs):
                log.info("%d/%d done %s (window utilization %s)", n, len(jobs), counts, rl.get("utilization"))

    try:
        await asyncio.gather(*(one(j) for j in jobs))
    finally:
        out.close()
    return counts

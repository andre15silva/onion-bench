import asyncio
import json
from datetime import date, datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from onion_bench import metrics as M
from onion_bench.assemble import DATASET_COLUMNS, PROMPT_FIELDS
from onion_bench.extract import ExtractConfig, extract
from onion_bench.leaks import find_leaks, is_blocking, rules_from_sources
from onion_bench.prompts import SYSTEM_PROMPT, parse_answer, user_message
from onion_bench.runner import EvalConfig, ModelSpec, plan_jobs, run_async
from onion_bench.scrape import canonical_url, load_sources, stratified_order
from onion_bench.discover import Candidate

HTML = """<html lang="en"><head>
<meta property="og:site_name" content="Example Times">
<meta property="og:title" content="OG title - Example Times">
<script type="application/ld+json">{"@graph":[{"@type":"NewsArticle","headline":"LD headline",
 "datePublished":"2026-08-01T10:00:00+00:00","author":[{"@type":"Person","name":"A. Writer"}],
 "articleSection":["Local"],"keywords":"a, b"}]}</script></head>
<body><p class="lead">The lead paragraph.</p><h1>Man  Does <em>Thing</em></h1>
<div class="body"><p>First <a href="#">linked</a> words, with comma.</p>
<figure><p>caption text</p></figure><p>Second<br>line.</p><p>Subscribe now</p></div></body></html>"""


def test_extract_verbatim_and_metadata():
    cfg = ExtractConfig(body_selector=".body", lead_selector=".lead", exclude_selectors=["figure"],
                        drop_paragraphs=["Subscribe now"], min_words_selector=1)
    ex = extract(HTML, cfg)
    assert ex.title == "Man Does Thing"
    assert ex.paragraphs == ["The lead paragraph.", "First linked words, with comma.", "Second line."]
    assert ex.published_at == datetime(2026, 8, 1, 10, tzinfo=timezone.utc)
    assert ex.authors == ["A. Writer"] and ex.section == "Local" and ex.keywords == ["a", "b"]
    assert ex.method == "selector"


def test_extract_text_date_fallback():
    html = "<html><body><h1>T</h1><time>August 28, 2026</time><div class='b'><p>" + "word " * 30 + "</p></div></body></html>"
    ex = extract(html, ExtractConfig(body_selector=".b", date_selector="time"))
    assert ex.published_at == datetime(2026, 8, 28, tzinfo=timezone.utc)


def test_leak_rules_from_config():
    rules = rules_from_sources(load_sources())
    hits = find_leaks("Ronaldo volta", "Mas o Imprensa Falsa sabe que o IF tem razão.", rules)
    assert {h["source"] for h in hits} == {"imprensa_falsa"}
    assert is_blocking(hits)
    assert find_leaks("Local man tells the BBC", "", rules)[0]["source"] == "bbc_news"
    # lower-case common words must not trip outlet patterns
    assert not find_leaks("If the observer sees an onion", "if if if", rules)
    # satire mentioned in body text is informational, not blocking
    hint = find_leaks("Comedian on tour", "His satire show sold out.", rules)
    assert hint and not is_blocking(hint)


def test_prompt_contains_only_title_and_text():
    assert PROMPT_FIELDS == ("title", "text")
    row = {c: f"<{c}-SENTINEL>" for c in DATASET_COLUMNS}
    row.update(title="Headline here", text="Body here.", url="https://theonion.com/x", source="the_onion",
               source_name="The Onion", authors=["The Onion Staff"])
    msg = user_message(row)
    assert "Headline here" in msg and "Body here." in msg
    for leaked in ("SENTINEL", "theonion", "Onion", "Staff"):
        assert leaked not in msg and leaked not in SYSTEM_PROMPT


@pytest.mark.parametrize("text,expected", [
    ('{"verdict": "fake", "p_fake": 87}', ("fake", 0.87)),
    ('Sure: {"verdict":"real","p_fake":5}', ("real", 0.05)),
    ('{"verdict": "fake", "p_fake": 140}', ("fake", 1.0)),
    ('{"verdict": "maybe", "p_fake": 50}', (None, None)),
    ("no json", (None, None)),
])
def test_parse_answer(text, expected):
    assert parse_answer(text) == expected


def test_canonical_url_and_stratified_order():
    assert canonical_url("https://WWW.bbc.co.uk/news/articles/abc?at_medium=RSS#x") == "https://www.bbc.co.uk/news/articles/abc"
    cands = [Candidate(f"u{m}-{i}", datetime(2026, m, 1, tzinfo=timezone.utc)) for m in (7, 8, 9) for i in range(3)]
    first3 = stratified_order(cands, seed=1)[:3]
    assert sorted(c.date_hint.month for c in first3) == [7, 8, 9]


# ------------------------------------------------------------------ metrics
def test_confusion_metrics():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    yh = np.array([1, 1, 0, 0, 0, 1, 0, 1])
    m = M.from_confusion(M.confusion(y, yh))
    assert m["recall"] == 0.75 and m["specificity"] == 0.75 and m["accuracy"] == 0.75
    assert m["mcc"] == pytest.approx(0.5)
    assert m["balanced_accuracy"] == pytest.approx(0.75)


def test_auroc_and_ap():
    y = np.array([0, 0, 1, 1])
    assert M.auroc(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert M.auroc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert M.average_precision(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx((1 + 2 / 3) / 2)


def test_mcnemar_and_kappa():
    assert M.mcnemar_exact(0, 0) == 1.0
    assert M.mcnemar_exact(10, 0) == pytest.approx(2 / 2**10)
    assert M.fleiss_kappa([5, 5, 5], [5, 0, 5]) == pytest.approx(1.0)


def test_evaluate_item_majority_and_strict():
    labels = {"a": 1, "b": 0, "c": 1}
    resp = (
        [{"item_id": "a", "verdict": v, "p_fake": p} for v, p in [("fake", .9), ("fake", .8), ("real", .4)]]
        + [{"item_id": "b", "verdict": "real", "p_fake": .1}] * 3
        + [{"item_id": "c", "verdict": None, "p_fake": None}] * 3
    )
    g = M.group_by_item(resp, labels)
    res = M.evaluate(g, ["a", "b", "c"], n_boot=0)
    assert res["n_calls"] == 9 and res["invalid_rate"] == pytest.approx(1 / 3)
    assert res["item_majority"]["n"] == 2 and res["item_majority"]["accuracy"] == 1.0
    assert res["call"]["n"] == 6 and res["call_strict"]["n"] == 9
    assert res["call_strict"]["fn"] == 1 + 3


# ------------------------------------------------------------------- runner
class FakeMessages:
    def __init__(self):
        self.calls = []

    async def create(self, **kw):
        self.calls.append(kw)
        assert "tools" not in kw and "fallbacks" not in kw and "temperature" not in kw
        if "REFUSE" in kw["messages"][0]["content"]:
            return SimpleNamespace(content=[], stop_reason="refusal", model=kw["model"], stop_details=SimpleNamespace(category="cyber"),
                                   usage=SimpleNamespace(input_tokens=10, output_tokens=0))
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking="hmm"), SimpleNamespace(type="text", text='{"verdict":"fake","p_fake":70}')],
            stop_reason="end_turn", model=kw["model"], stop_details=None,
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


def test_runner_resume_cutoff_and_records(tmp_path):
    items = [
        {"id": "x1", "title": "T1", "text": "B1", "published_at": "2026-07-02T00:00:00+00:00"},
        {"id": "x2", "title": "REFUSE", "text": "B2", "published_at": "2026-08-02T00:00:00+00:00"},
        {"id": "old", "title": "T0", "text": "B0", "published_at": "2026-06-01T00:00:00+00:00"},
    ]
    cfg = EvalConfig(models=[ModelSpec("m-a", date(2026, 6, 30)), ModelSpec("m-b", date(2026, 7, 15))], n_samples=5, concurrency=3)
    out = tmp_path / "responses.jsonl"
    jobs = plan_jobs(cfg, items, out)
    assert len(jobs) == 5 * 2 + 5 * 1  # "old" skipped for both, x1 skipped for m-b
    client = SimpleNamespace(messages=FakeMessages())
    counts = asyncio.run(run_async(cfg, jobs, out, client=client))
    assert counts == {"ok": 5, "refusal": 10}
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert all(r["prompt_version"] for r in recs)
    assert {r["sample"] for r in recs} == set(range(5))
    assert plan_jobs(cfg, items, out) == []  # fully resumed
    call = client.messages.calls[0]
    assert call["thinking"]["type"] == "adaptive"
    assert call["output_config"]["format"]["type"] == "json_schema"


def test_no_generic_fallback_when_body_container_is_empty():
    html = "<html><body><h1>Headline Only</h1><div class='body'></div><ul>" + "".join(
        f"<li><a href='/x{i}'>Some Other Satirical Headline Number {i} Here</a></li>" for i in range(20)) + "</ul></body></html>"
    ex = extract(html, ExtractConfig(body_selector=".body"))
    assert ex.paragraphs == [] and ex.method == "selector"


def test_balance_items_per_language():
    from onion_bench.runner import balance_items

    items = (
        [{"id": f"f{i}", "language": "pt", "label": "fake", "subset": "satire", "source": "if"} for i in range(3)]
        + [{"id": f"r{i}", "language": "pt", "label": "real", "subset": "real_control", "source": "rtp" if i % 2 else "obs"} for i in range(10)]
        + [{"id": "imp", "language": "pt", "label": "real", "subset": "real_implausible", "source": "x"}]
        + [{"id": f"e{i}", "language": "en", "label": "fake", "subset": "satire", "source": "onion"} for i in range(2)]
    )
    out = balance_items(items, seed=0)
    pt = [i for i in out if i["language"] == "pt"]
    assert sum(i["label"] == "fake" for i in pt) == 3 and sum(i["label"] == "real" for i in pt) == 3
    assert any(i["id"] == "imp" for i in out)  # implausible kept first
    assert {i["source"] for i in pt if i["label"] == "real"} == {"x", "obs", "rtp"}  # round-robin
    assert not [i for i in out if i["language"] == "en"]  # no real en -> nothing to compare
    assert out == balance_items(items, seed=0)


def _stream(tools=("StructuredOutput",), api_key="none", served="m-a", so=None, stop="tool_use", is_error=False, text=None):
    ev = [
        {"type": "system", "subtype": "init", "tools": list(tools), "mcp_servers": [], "apiKeySource": api_key, "model": served},
        {"type": "assistant", "message": {"model": served, "content": []}},
        {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", "rateLimitType": "five_hour", "resetsAt": 1,
                                                          "unifiedWindows": {"five_hour": {"utilization": 0.3}}}},
        {"type": "result", "structured_output": so, "stop_reason": stop, "is_error": is_error, "result": text,
         "modelUsage": {served: {}}, "usage": {"input_tokens": 2, "cache_creation_input_tokens": 1000, "output_tokens": 50}, "duration_ms": 1500},
    ]
    return [json.dumps(e) for e in ev]


def test_cli_backend_parse_stream():
    from onion_bench.cli_backend import command, parse_stream
    from onion_bench.runner import Job

    job = Job(ModelSpec("m-a", effort="high"), {"id": "x", "title": "T", "text": "B"}, 0)
    ok = parse_stream(job, _stream(so={"verdict": "fake", "p_fake": 80}))
    assert ok["status"] == "ok" and ok["verdict"] == "fake" and ok["p_fake"] == 0.8
    assert ok["backend"] == "claude_cli" and ok["rate_limit"]["utilization"] == 0.3 and ok["input_tokens"] == 1002
    assert parse_stream(job, _stream(tools=("StructuredOutput", "WebSearch"), so={"verdict": "fake", "p_fake": 1}))["status"] == "invalid_setup"
    assert parse_stream(job, _stream(api_key="ANTHROPIC_API_KEY", so={"verdict": "fake", "p_fake": 1}))["status"] == "invalid_setup"
    assert parse_stream(job, _stream(served="m-b", so={"verdict": "fake", "p_fake": 1}))["status"] == "invalid_setup"
    assert parse_stream(job, _stream(stop="refusal"))["status"] == "refusal"
    assert parse_stream(job, _stream(is_error=True, text="You've hit your session limit"))["status"] == "rate_limited"
    assert parse_stream(job, [])["status"] == "api_error"
    cmd = command(job)
    assert cmd[cmd.index("--tools") + 1] == "" and "--safe-mode" in cmd and cmd[cmd.index("--model") + 1] == "m-a"


def test_cli_child_env_drops_keys(monkeypatch):
    from onion_bench.cli_backend import child_env

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_USER_EMAIL", "x@y")
    env = child_env()
    assert "ANTHROPIC_API_KEY" not in env and "CLAUDE_CODE_USER_EMAIL" not in env and "PATH" in env


def test_stratified_sample_keeps_mix():
    from onion_bench.runner import stratified_sample

    items = [{"id": f"{l}{lab}{i}", "language": l, "label": lab, "subset": "s", "source": f"src{i % 3}"}
             for l, n in (("en", 60), ("pt", 20)) for lab in ("fake", "real") for i in range(n)]
    sub = stratified_sample(items, 40, seed=0)
    from collections import Counter
    c = Counter((i["language"], i["label"]) for i in sub)
    assert c == {("en", "fake"): 15, ("en", "real"): 15, ("pt", "fake"): 5, ("pt", "real"): 5}
    assert sub == stratified_sample(items, 40, seed=0)

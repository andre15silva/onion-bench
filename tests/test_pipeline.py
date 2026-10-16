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


def test_no_generic_fallback_when_disabled_and_container_missing():
    html = "<html><body><h1>Headline Only</h1><ul>" + "".join(
        f"<li>Some Other Satirical Headline Number {i} Here Today</li>" for i in range(20)) + "</ul></body></html>"
    assert extract(html, ExtractConfig(body_selector=".entry-content", generic_fallback=False)).paragraphs == []
    assert extract(html, ExtractConfig(body_selector=".entry-content")).method in ("trafilatura", "none")


def test_visually_hidden_text_excluded():
    html = ("<html><body><h1>T</h1><article><div data-testid='rich-text'><p>They <a href='#'>cut water use "
            "<span class='visually-hidden x'>, external</span></a>now, officials said in a long statement today.</p></div></article></body></html>")
    cfg = ExtractConfig(body_selector="article", paragraph_selector="[data-testid='rich-text'] p",
                        exclude_selectors=[".visually-hidden"], min_words_selector=1, generic_fallback=False)
    assert extract(html, cfg).paragraphs == ["They cut water use now, officials said in a long statement today."]


def test_style_swap_only_touches_style():
    from onion_bench.diagnostics import style_swap

    items = [
        {"id": "o", "source": "the_onion", "title": "Area Man Wins Big", "text": "ST. LOUIS—Area man won.\n\nMore."},
        {"id": "b", "source": "bbc_news", "title": "Man jailed for theft", "text": "A man was jailed."},
        {"id": "r", "source": "rtp_noticias", "title": "x", "text": "y"},
    ]
    o, b = style_swap(items)
    assert o["title"] == "Area man wins big" and o["text"] == "Area man won.\n\nMore."
    assert b["title"] == "Man Jailed for Theft" and b["text"] == "LONDON—A man was jailed."
    assert items[0]["title"] == "Area Man Wins Big"  # originals untouched


def test_lead_whole_sentences_verbatim():
    from onion_bench.prompts import lead

    text = ("WASHINGTON—Sen. John Q. Public said Tuesday the U.S. would act. He added more words here to pad it out a bit."
            "\n\nSecond paragraph sentence one. Sentence two.")
    assert lead(text, min_words=5) == "WASHINGTON—Sen. John Q. Public said Tuesday the U.S. would act."
    assert lead(text, min_words=25) == text.split("\n\n")[0] + "\n\nSecond paragraph sentence one."
    assert lead("“Quoted.” Then more.", min_words=1) == "“Quoted.”"
    # every lead is a verbatim prefix of the text (modulo the paragraph join)
    import json
    rows = [json.loads(l) for l in open("data/dataset/onion_bench.jsonl")][:300] if __import__("os").path.exists("data/dataset/onion_bench.jsonl") else []
    for r in rows:
        assert r["text"].startswith(lead(r["text"]))


def test_prompt_variants():
    from onion_bench.prompts import VARIANTS, system_prompt, user_message

    it = {"title": "Headline", "text": "One sentence here. " * 30}
    assert "<article>" in user_message(it, "full") and "<article_opening>" in user_message(it, "headline_lead")
    assert user_message(it, "headline_only") == "<headline>\nHeadline\n</headline>"
    assert all(system_prompt(v) != system_prompt("full") for v in VARIANTS if v != "full")
    with pytest.raises(ValueError):
        user_message(it, "nope")


def test_variant_is_recorded_and_not_mixed(tmp_path):
    from onion_bench.cli_backend import command
    from onion_bench.runner import Job

    items = [{"id": "x1", "title": "T1", "text": "Sentence one is here. " * 20, "published_at": "2026-07-02T00:00:00+00:00"}]
    cfg = EvalConfig(models=[ModelSpec("m-a", date(2026, 6, 30))], n_samples=1, prompt_variant="headline_only")
    out = tmp_path / "responses.jsonl"
    jobs = plan_jobs(cfg, items, out)
    assert jobs[0].variant == "headline_only"
    cmd = command(jobs[0])
    assert cmd[cmd.index("-p") + 1] == "<headline>\nT1\n</headline>"
    assert "shown the headline" in cmd[cmd.index("--system-prompt") + 1]
    asyncio.run(run_async(cfg, jobs, out, client=SimpleNamespace(messages=FakeMessages())))
    assert json.loads(out.read_text().splitlines()[0])["prompt_variant"] == "headline_only"
    cfg.prompt_variant = "full"
    with pytest.raises(ValueError):
        plan_jobs(cfg, items, out)


def test_import_index_strips_branding(tmp_path):
    from onion_bench.scrape import get_source, import_index

    src = get_source("inimigo_publico")
    page = tmp_path / "index.html"
    page.write_text(
        "<html><body>"
        "<h2><a href='https://expresso.pt/inimigo-publico/2026-08-01-inimigo-publico-x-ab12?utm_source=y'>Inimigo Público: Ministro faz coisa</a></h2>"
        "<a href='https://expresso.pt/inimigo-publico/2026-08-02-y-cd34'><h3>Inimigo Público. Outra coisa</h3></a>"
        "<h3><a href='https://expresso.pt/inimigo-publico/2026-08-03-cartoon-ef56'>Cartoon*</a></h3>"
        "</body></html>", encoding="utf-8")
    n, urls = import_index(src, page, tmp_path, src["index_url_pattern"])
    rows = [json.loads(l) for l in (tmp_path / "inimigo_publico.jsonl").read_text().splitlines()]
    assert n == 2 and len(urls) == 2
    assert [r["title"] for r in rows] == ["Ministro faz coisa", "Outra coisa"]
    assert rows[0]["original_title"] == "Inimigo Público: Ministro faz coisa"
    assert rows[0]["url"] == "https://expresso.pt/inimigo-publico/2026-08-01-inimigo-publico-x-ab12"
    assert rows[0]["published_at"].startswith("2026-08-01") and rows[0]["headline_only"]


def test_headline_only_items_only_in_headline_variant(tmp_path):
    from onion_bench.runner import load_items

    ds = tmp_path / "d.jsonl"
    ds.write_text("\n".join(json.dumps(r) for r in [
        {"id": "a", "eval_eligible": True, "has_text": True},
        {"id": "b", "eval_eligible": True, "has_text": False},
    ]))
    assert [r["id"] for r in load_items(ds)] == ["a"]
    assert [r["id"] for r in load_items(ds, prompt_variant="headline_only")] == ["a", "b"]


def _claim_page(claim, verdict, author=None, item_date=None, review_date="2026-08-10T10:00:00Z"):
    item = {"@type": "CreativeWork"}
    if author:
        item["author"] = {"@type": "Person", "name": author}
    if item_date:
        item["datePublished"] = item_date
    ld = {"@context": "https://schema.org", "@type": "ClaimReview", "claimReviewed": claim, "datePublished": review_date,
          "reviewRating": {"@type": "Rating", "alternateName": verdict}, "itemReviewed": item}
    return f"<html><head><script type='application/ld+json'>{json.dumps(ld)}</script></head><body></body></html>"


def test_claim_review_records():
    from onion_bench.factcheck import build_claim_record
    from onion_bench.scrape import get_source

    src = dict(get_source("observador_factcheck"), _since=datetime(2026, 7, 1, tzinfo=timezone.utc))
    r = build_claim_record(src, "https://observador.pt/factchecks/x/", _claim_page("Governo proíbe carros a gasóleo.", "Errado", "Utilizador de Facebook"))
    assert (r["label"], r["subset"], r["reject"], r["sourcing_mode"]) == ("fake", "misinformation", None, "fact_check_claimreview")
    assert r["headline_only"] and r["text"] == "" and r["provenance"]["verdict"] == "Errado"
    assert build_claim_record(src, "u", _claim_page("Algo verdadeiro.", "Certo"))["subset"] == "fact_checked_true"
    assert build_claim_record(src, "u", _claim_page("x", "Esticado"))["reject"] == "rating(Esticado)"
    assert build_claim_record(src, "u", _claim_page("Disse isto.", "Errado", "Luís Montenegro"))["reject"] == "attributed_statement"
    assert build_claim_record(src, "u", _claim_page("Vídeo mostra cheias.", "Errado"))["reject"] == "media_only_claim"
    assert build_claim_record(src, "u", _claim_page("Velho boato.", "Errado", item_date="2019-01-01"))["reject"] == "claim_before_since"


def test_counterfactual_apply_and_checks():
    from onion_bench.counterfactual import apply_replacements, check
    from onion_bench.leaks import rules_from_sources

    base = {"title": "Council approves 12 new homes in Leeds",
            "text": "A council has approved plans for 12 new homes in Leeds. " + "The plans were discussed by members. " * 10 + "A final vote happened late on Tuesday."}
    reps = [{"field": "title", "find": "12 new homes", "replace": "120 new homes"},
            {"field": "text", "find": "12 new homes", "replace": "120 new homes"}]
    t, x, applied = apply_replacements(base["title"], base["text"], reps)
    assert t == "Council approves 120 new homes in Leeds" and x.startswith("A council has approved plans for 120 new homes")
    c = check(base, t, x, applied, rules_from_sources(load_sources()))
    assert all(c.values()), c
    # an edit only deep in the body fails the opening/title checks
    t2, x2, a2 = apply_replacements(base["title"], base["text"], [{"field": "text", "find": "late on Tuesday", "replace": "early on Wednesday"}])
    c2 = check(base, t2, x2, a2, rules_from_sources(load_sources()))
    assert not c2["title_changed"] and not c2["opening_changed"]
    # a find that doesn't match, and an outlet name introduced
    _, _, a3 = apply_replacements(base["title"], base["text"], [{"field": "title", "find": "nope", "replace": "x"}])
    assert not check(base, base["title"], base["text"], a3, rules_from_sources(load_sources()))["all_finds_matched"]
    t4, x4, a4 = apply_replacements(base["title"], base["text"], [{"field": "title", "find": "Council", "replace": "BBC"}])
    assert not check(base, t4, x4, a4, rules_from_sources(load_sources()))["no_new_outlet_mentions"]


def test_select_bases_round_robin():
    from onion_bench.counterfactual import select_bases

    items = [{"id": f"{s}{i}", "language": "en", "subset": "real_control", "has_text": True, "eval_eligible": True, "source": s}
             for s in ("bbc", "cbs") for i in range(5)] + [{"id": "sat", "language": "en", "subset": "satire", "has_text": True, "eval_eligible": True, "source": "on"}]
    b = select_bases(items, {"en": 4})
    assert len(b) == 4 and {x["source"] for x in b} == {"bbc", "cbs"}


def test_balance_keeps_counterfactual_bases():
    from onion_bench.runner import balance_items

    reals = [{"id": f"r{i}", "language": "pt", "label": "real", "subset": "real_control", "source": "rtp"} for i in range(20)]
    cfs = [{"id": "cf1", "language": "pt", "label": "fake", "subset": "counterfactual", "source": "counterfactual", "provenance": {"base_id": "r17"}},
           {"id": "cf2", "language": "pt", "label": "fake", "subset": "counterfactual", "source": "counterfactual", "provenance": {"base_id": "r3"}}]
    out = {i["id"] for i in balance_items(reals + cfs, seed=0)}
    assert {"cf1", "cf2", "r17", "r3"} <= out and len(out) == 4


def test_report_per_source_splits_subsets(tmp_path):
    from onion_bench.report import build_report

    items = [
        {"id": "f", "language": "en", "label": "fake", "subset": "misinformation", "source": "snopes_claims", "published_at": "2026-08-01T00:00:00+00:00"},
        {"id": "t", "language": "en", "label": "real", "subset": "fact_checked_true", "source": "snopes_claims", "published_at": "2026-08-01T00:00:00+00:00"},
    ]
    resp = tmp_path / "responses.jsonl"
    resp.write_text("\n".join(json.dumps({"model": "m", "item_id": i, "sample": 0, "status": "ok", "verdict": v, "p_fake": p})
                              for i, v, p in [("f", "fake", 0.9), ("t", "fake", 0.8)]))
    cfg = EvalConfig(models=[ModelSpec("m", date(2026, 6, 30))])
    out = build_report(cfg, items, resp, tmp_path, n_boot=0)
    ps = out["models"]["m"]["per_source"]
    assert ps["snopes_claims/misinformation"]["judged_fake_rate_calls"] == 1.0
    assert ps["snopes_claims/fact_checked_true"]["judged_fake_rate_calls"] == 1.0 and len(ps) == 2

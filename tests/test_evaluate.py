"""Milestone 4: evaluation metrics."""
import pytest

from app.evaluate import aggregate, aspect_f1, lead_1, rouge_l, rouge_n, score, to_markdown

REC = {"feedback": "The labs were fun. The course moved way too fast. I am a part-time student.",
       "summary": "Overall: mixed. Praises: labs. Concerns: pacing.",
       "aspects": [{"aspect": "labs", "sentiment": "pos"}, {"aspect": "pacing", "sentiment": "neg"}]}


def test_rouge_known_values():
    assert rouge_n("the cat sat", "the cat sat") == 1.0
    assert rouge_n("a b", "c d") == 0.0
    assert rouge_n("the cat", "the cat sat on") == pytest.approx(2 / 3)       # p=1, r=0.5
    assert rouge_n("the cat sat", "the cat ran", 2) == pytest.approx(0.5)
    assert rouge_l("a b c d", "a x c d") == pytest.approx(0.75)               # LCS = a c d
    assert rouge_l("", "a") == 0.0


def test_aspect_f1_rewards_right_aspects_with_right_sentiment():
    assert aspect_f1(REC["summary"], REC) == 1.0
    assert aspect_f1("Overall: mixed. Praises: pacing. Concerns: labs.", REC) == 0.0   # sentiments swapped
    assert aspect_f1("Overall: positive. Praises: labs.", REC) == pytest.approx(2 / 3)  # one of two
    assert aspect_f1("The students liked it.", REC) == 0.0


def test_perfect_summary_scores_perfectly_and_is_concise():
    s = score(REC["summary"], REC)
    assert s["rouge1"] == s["rougeL"] == s["aspect_f1"] == s["overall_acc"] == s["format_ok"] == 1.0
    assert s["compression"] < 0.6


def test_lead_baseline_and_report():
    assert lead_1(REC["feedback"]) == "The labs were fun."
    scores = [score(lead_1(REC["feedback"]), REC), score(REC["summary"], REC)]
    agg = aggregate(scores)
    assert agg["format_ok"] == 0.5
    table = to_markdown({"x": {**agg, "ms_per_summary": 0}}, 2)
    assert "| x | " in table and "aspect_f1" in table

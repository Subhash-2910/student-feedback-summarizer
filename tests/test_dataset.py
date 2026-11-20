"""Milestone 1: the synthetic dataset."""
from app.dataset import ASPECTS, PHRASES, generate, load_jsonl, make_summary, save_jsonl


def test_generation_is_reproducible():
    assert generate(50, 10, seed=1) == generate(50, 10, seed=1)
    assert generate(50, 10, seed=1) != generate(50, 10, seed=2)


def test_examples_are_diverse():
    train, _ = generate(400, 60)
    assert len({r["feedback"] for r in train}) > 380          # almost no duplicate comments
    assert len({r["summary"] for r in train}) > 100           # many different targets
    counts = {a: sum(a in {x["aspect"] for x in r["aspects"]} for r in train) for a in ASPECTS}
    assert min(counts.values()) >= 40                          # every aspect is well covered
    overalls = {r["summary"].split(".")[0] for r in train}
    assert overalls == {"Overall: positive", "Overall: negative", "Overall: mixed"}


def test_summary_matches_labels():
    assert make_summary([("pacing", "neg"), ("lectures", "pos"), ("labs", "pos")]) == \
        "Overall: mixed. Praises: lectures, labs. Concerns: pacing."
    assert make_summary([("grading", "pos")]) == "Overall: positive. Praises: grading."
    train, test = generate(100, 20)
    for r in train + test:
        assert r["summary"] == make_summary([(a["aspect"], a["sentiment"]) for a in r["aspects"]])


def test_test_split_uses_unseen_phrasings():
    train, test = generate(400, 60)
    held_out = [pool[-1].lower() for by_sent in PHRASES.values() for pool in by_sent.values()]
    train_text = " ".join(r["feedback"].lower() for r in train)
    assert not any(p in train_text for p in held_out)
    assert all(any(p in r["feedback"].lower() for p in held_out) for r in test)


def test_jsonl_round_trip(tmp_path):
    train, _ = generate(5, 1)
    save_jsonl(train, tmp_path / "x.jsonl")
    assert load_jsonl(tmp_path / "x.jsonl") == train

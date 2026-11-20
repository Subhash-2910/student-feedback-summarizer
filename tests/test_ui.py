"""Milestone 3: the Streamlit UI and the helpers it uses to read uploads."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.summarize import parse_summary, split_feedback, tally


def test_csv_upload_uses_the_feedback_column():
    csv = b"student,Feedback\n1,Labs were great.\n2,\n3,Too fast.\n"
    assert split_feedback(csv, "survey.csv") == ["Labs were great.", "Too fast."]


def test_csv_without_known_column_uses_the_first():
    assert split_feedback("notes\nGood labs\n", "x.csv") == ["Good labs"]


def test_text_is_split_by_line_or_paragraph():
    assert split_feedback("one\n\ntwo\n") == ["one", "two"]
    assert split_feedback("first comment\nsecond comment") == ["first comment", "second comment"]
    assert split_feedback("a long comment\nthat wraps\n\nanother one") == ["a long comment that wraps", "another one"]


def test_parse_and_tally_summaries():
    s = "Overall: mixed. Praises: lectures, labs. Concerns: pacing."
    assert parse_summary(s) == {"overall": "mixed", "praises": ["lectures", "labs"], "concerns": ["pacing"]}
    assert parse_summary("the course was nice") == {"overall": None, "praises": [], "concerns": []}
    rows = tally([s, "Overall: negative. Concerns: pacing, grading."])
    assert rows[0] == {"aspect": "pacing", "praised": 0, "concerns": 2}
    assert {r["aspect"] for r in rows} == {"lectures", "labs", "pacing", "grading"}


def test_page_renders_without_errors():
    at = AppTest.from_file(str(Path(__file__).parent.parent / "app" / "main.py"), default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Student Feedback Summarizer"
    assert at.button[0].label == "Summarize"
    assert any(m.value == "**3** comment(s) found." for m in at.markdown)   # the example text

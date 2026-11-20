"""Milestone 1: a synthetic dataset of student feedback and summary pairs.

Each record is one student's free-text comment about a course. It touches 1-4 *aspects* of the course
(lectures, pacing, grading, ...), each with a positive or negative sentiment, plus some off-topic filler.
The target summary is a short, structured line built from those labels:

    Overall: mixed. Praises: lectures, labs. Concerns: pacing.

Because we keep the labels, we can later check whether a generated summary names the right aspects
(see app/evaluate.py), not just whether it shares words with the reference.

Diversity comes from: 10 aspects x 2 sentiments x 5 phrasings, 12 openers, 12 filler sentences,
random order, random number of aspects, typos/lowercase noise and 8 course names.
The last phrasing of every (aspect, sentiment) pair is *held out*: it only appears in the test split,
so the test measures whether the model learned the task rather than memorised sentences.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ASPECTS = ["lectures", "assignments", "pacing", "instructor", "course materials",
           "labs", "exams", "grading", "workload", "office hours"]

PHRASES: dict[str, dict[str, list[str]]] = {
    "lectures": {
        "pos": ["the lectures were clear and well organised",
                "I really enjoyed the lectures, they were engaging",
                "lectures explained hard ideas with great examples",
                "the lecture videos were easy to follow",
                "every lecture felt worth attending"],
        "neg": ["the lectures were hard to follow",
                "lectures were mostly reading slides out loud",
                "I found the lectures boring and unstructured",
                "the lecture recordings had terrible audio",
                "lectures often ran over time and felt rushed at the end"],
    },
    "assignments": {
        "pos": ["the assignments were challenging but fair",
                "homework problems really helped me practise",
                "the projects were fun and relevant to real work",
                "assignments built nicely on each other",
                "I learned the most from the weekly problem sets"],
        "neg": ["the assignments had unclear instructions",
                "homework felt like busywork",
                "the project spec kept changing after release",
                "assignments did not match what was taught in class",
                "the problem sets were full of typos and errors"],
    },
    "pacing": {
        "pos": ["the pace of the course was just right",
                "topics were introduced at a comfortable speed",
                "the course never felt rushed",
                "the schedule gave us enough time to absorb each topic",
                "the progression from basics to advanced material was smooth"],
        "neg": ["the course moved way too fast",
                "we rushed through the second half",
                "the pacing was uneven, slow at first then frantic",
                "there was not enough time to digest each topic",
                "too many topics were crammed into the final weeks"],
    },
    "instructor": {
        "pos": ["the instructor was enthusiastic and approachable",
                "the professor clearly cared about our learning",
                "our teacher was patient with questions",
                "the instructor explained things with real passion",
                "the lecturer made everyone feel welcome to ask questions"],
        "neg": ["the instructor seemed disengaged",
                "the professor was dismissive of questions",
                "our teacher often came unprepared",
                "the instructor rarely answered emails",
                "the lecturer was condescending when students made mistakes"],
    },
    "course materials": {
        "pos": ["the slides and notes were excellent",
                "the textbook was a great companion",
                "the reading materials were well chosen",
                "the online resources were really helpful",
                "the handouts summarised each week perfectly"],
        "neg": ["the slides were outdated",
                "the textbook was expensive and barely used",
                "the reading list was overwhelming",
                "the course notes had many mistakes",
                "the materials on the website were disorganised and hard to find"],
    },
    "labs": {
        "pos": ["the lab sessions were very practical",
                "the hands-on labs were my favourite part",
                "labs helped me connect theory to practice",
                "the TAs in the lab were super helpful",
                "lab exercises were well designed and fun"],
        "neg": ["the labs were too crowded",
                "lab equipment often did not work",
                "the lab sessions felt disconnected from the lectures",
                "labs were too short to finish the exercises",
                "the lab setup instructions never worked on my laptop"],
    },
    "exams": {
        "pos": ["the exams were fair and reflected the material",
                "the midterm tested what we actually practised",
                "exam questions were clear",
                "the practice exams prepared us well",
                "the final exam was tough but reasonable"],
        "neg": ["the exams were unreasonably hard",
                "the final had questions on topics we never covered",
                "exam time was far too short",
                "the midterm questions were confusing",
                "the exams rewarded memorisation instead of understanding"],
    },
    "grading": {
        "pos": ["grading was transparent and fair",
                "feedback on graded work was detailed",
                "the rubric made expectations clear",
                "grades came back quickly",
                "the grading scheme rewarded steady effort"],
        "neg": ["grading felt arbitrary",
                "grades took weeks to come back",
                "there was almost no feedback on graded work",
                "the rubric was never shared",
                "grading was inconsistent between TAs"],
    },
    "workload": {
        "pos": ["the workload was manageable",
                "the amount of work was reasonable for the credits",
                "I could balance this course with my other classes",
                "weekly effort was predictable and fair",
                "the workload was spread evenly across the term"],
        "neg": ["the workload was overwhelming",
                "this course took far more hours than it should",
                "there was too much work every week",
                "deadlines piled up with no breathing room",
                "the workload made it impossible to keep up with other courses"],
    },
    "office hours": {
        "pos": ["office hours were really useful",
                "TAs in office hours were patient and helpful",
                "there were plenty of office hour slots",
                "I always got my questions answered in office hours",
                "office hours were friendly and well run"],
        "neg": ["office hours were always packed",
                "there were too few office hours",
                "office hours were at inconvenient times",
                "office hours were often cancelled without notice",
                "the queue at office hours was so long I never got help"],
    },
}

OPENERS = ["", "Honestly,", "Overall,", "To be fair,", "In my opinion,", "I think", "I felt that",
           "Also,", "On the other hand,", "Personally,", "One thing:", "I noticed that"]

FILLER = ["I took this course as an elective.", "This was my second semester at the university.",
          "I am a part-time student.", "I mostly attended online.", "I needed this course for my major.",
          "My friend recommended this class.", "I sat near the back most weeks.",
          "I had some background in the subject already.", "This was my first course in the department.",
          "I usually study in the library.", "I joined the course a week late.",
          "I am writing this after the final."]

COURSES = ["Intro to Programming", "Linear Algebra", "Organic Chemistry", "World History",
           "Data Structures", "Microeconomics", "Statistics 101", "Machine Learning"]

PROMPT = "Student feedback: {feedback}\nSummary:"


def make_summary(aspects: list[tuple[str, str]]) -> str:
    """The reference summary for a list of (aspect, 'pos'|'neg') labels, in a fixed canonical order."""
    pos = [a for a in ASPECTS if (a, "pos") in aspects]
    neg = [a for a in ASPECTS if (a, "neg") in aspects]
    overall = "mixed" if pos and neg else ("positive" if pos else "negative")
    parts = [f"Overall: {overall}."]
    if pos:
        parts.append("Praises: " + ", ".join(pos) + ".")
    if neg:
        parts.append("Concerns: " + ", ".join(neg) + ".")
    return " ".join(parts)


def _sentence(rng: random.Random, phrase: str) -> str:
    opener = rng.choice(OPENERS)
    text = f"{opener} {phrase}" if opener else phrase
    text = text[0].upper() + text[1:]
    end = rng.choice([".", ".", ".", "!"])
    if rng.random() < 0.1:  # informal writing: all lowercase
        text = text.lower()
    return text + end


def make_record(rng: random.Random, idx: int, holdout: bool) -> dict:
    """One synthetic comment. holdout=True uses only the held-out (last) phrasing of each label."""
    n = rng.choices([1, 2, 3, 4], weights=[2, 4, 3, 1])[0]
    chosen = rng.sample(ASPECTS, n)
    labels = [(a, rng.choice(["pos", "neg"])) for a in chosen]
    sentences = []
    for aspect, sentiment in labels:
        pool = PHRASES[aspect][sentiment]
        phrase = pool[-1] if holdout else rng.choice(pool[:-1])
        sentences.append(_sentence(rng, phrase))
    for _ in range(rng.choice([0, 0, 1, 1, 2])):
        sentences.insert(rng.randrange(len(sentences) + 1), rng.choice(FILLER))
    return {
        "id": f"{'test' if holdout else 'train'}-{idx:04d}",
        "course": rng.choice(COURSES),
        "feedback": " ".join(sentences),
        "summary": make_summary(labels),
        "aspects": [{"aspect": a, "sentiment": s} for a, s in labels],
    }


def generate(n_train: int = 400, n_test: int = 60, seed: int = 13) -> tuple[list[dict], list[dict]]:
    """Return (train, test). Same seed -> same data. Test uses held-out phrasings only."""
    rng = random.Random(seed)
    train = [make_record(rng, i, holdout=False) for i in range(n_train)]
    test = [make_record(rng, i, holdout=True) for i in range(n_test)]
    return train, test


def save_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    tr, te = generate()
    save_jsonl(tr, "dataset/train.jsonl")
    save_jsonl(te, "dataset/test.jsonl")
    print(f"wrote {len(tr)} train and {len(te)} test examples to dataset/")
    print(json.dumps(tr[0], indent=2))

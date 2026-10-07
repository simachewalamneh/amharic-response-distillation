"""Blind rating app (Streamlit).

Put this file in the human_eval folder, next to rating_items.json, then run:
    streamlit run rate_app.py
Ratings are written to ratings.csv in the same folder, in the format score.py reads.
The app never reads rating_key.csv, so you cannot see which model wrote which answer.
"""
import csv
import json
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
ITEMS_PATH, OUT_PATH = HERE / "rating_items.json", HERE / "ratings.csv"
FIELDS = ["fluency_A", "correct_A", "fluency_B", "correct_B"]
LABEL = {"fluency": "Fluency", "correct": "Correct"}

st.set_page_config(page_title="Blind rating", layout="wide")

if not ITEMS_PATH.exists():
    st.error(f"Cannot find {ITEMS_PATH.name} next to this script ({HERE}). "
             "Put rate_app.py in the human_eval folder and start it again.")
    st.stop()

items = json.loads(ITEMS_PATH.read_text(encoding="utf-8"))
n = len(items)


def complete(eid):
    r = st.session_state.ratings.get(eid, {})
    return all(r.get(f) for f in FIELDS)


def save():
    rows = [[it["eval_id"], *[st.session_state.ratings[it["eval_id"]][f] for f in FIELDS]]
            for it in items if complete(it["eval_id"])]
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eval_id", *FIELDS])
        w.writerows(rows)


def load_saved():
    saved = {}
    if OUT_PATH.exists():
        with OUT_PATH.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                saved[row["eval_id"]] = {k: int(row[k]) for k in FIELDS if row.get(k)}
    return saved


if "ratings" not in st.session_state:
    st.session_state.ratings = load_saved()
    first_open = next((i for i, it in enumerate(items) if not complete(it["eval_id"])), 0)
    st.session_state.idx = first_open
    st.session_state.jump = first_open + 1


def rate(eid, field):
    st.session_state.ratings.setdefault(eid, {})[field] = st.session_state[f"{eid}_{field}"]
    save()


def go(i):
    st.session_state.idx = max(0, min(n - 1, i))
    st.session_state.jump = st.session_state.idx + 1


idx = st.session_state.idx
it = items[idx]
eid = it["eval_id"]
done = sum(1 for x in items if complete(x["eval_id"]))

with st.sidebar:
    st.progress(done / n, text=f"{done} of {n} items rated")
    st.number_input("Go to item", min_value=1, max_value=n, step=1, key="jump",
                    on_change=lambda: go(st.session_state.jump - 1))
    todo = [str(i + 1) for i, x in enumerate(items) if not complete(x["eval_id"])]
    st.caption("Not yet rated: " + (", ".join(todo) if todo else "none"))
    with st.expander("Rubric", expanded=False):
        st.markdown(
            "**Fluency**: is it good text in the target language? "
            "1 garbled or wrong language, 3 understandable with clear errors, 5 natural.\n\n"
            "**Correct**: is the content right for the task? "
            "1 wrong, unrelated or empty, 3 partly right, 5 fully right.\n\n"
            "- Knowledge: facts must be right. If you cannot verify a claim, do not give above 3.\n"
            "- Instructions: answers the question asked, correctly, from the passage.\n"
            "- Translation: meaning preserved in the target language.\n"
            "- Summary: faithful to the passage, main point covered, nothing invented.\n\n"
            "Read each whole answer, including the end. Rate fluency and correctness separately, "
            "and rate each answer on its own before comparing.")
    st.caption(f"An item is saved to ratings.csv once all four ratings are set. File: {OUT_PATH}")

st.subheader(f"Item {idx + 1} of {n}: {it['task'].replace('_', ' ')}")
st.text_area("Prompt", it["prompt"], height=220, disabled=True, key=f"prompt_{eid}")

for col, side in zip(st.columns(2), "AB"):
    with col:
        st.markdown(f"#### Answer {side}")
        st.text_area(f"Answer {side}", it[side] or "(empty answer)", height=320, disabled=True,
                     key=f"text_{eid}_{side}", label_visibility="collapsed")
        for kind in ("fluency", "correct"):
            field = f"{kind}_{side}"
            cur = st.session_state.ratings.get(eid, {}).get(field)
            st.radio(LABEL[kind], [1, 2, 3, 4, 5], index=None if cur is None else cur - 1,
                     horizontal=True, key=f"{eid}_{field}", on_change=rate, args=(eid, field))

left, right, _ = st.columns([1, 1, 4])
left.button("Previous", on_click=go, args=(idx - 1,), disabled=idx == 0)
right.button("Next", on_click=go, args=(idx + 1,), disabled=idx == n - 1, type="primary")

if done == n:
    st.success(f"All {n} items rated and saved to {OUT_PATH.name}. Stop the app (Ctrl+C in the terminal) "
               "and run: python3 score.py")

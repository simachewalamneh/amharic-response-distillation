# Blind human evaluation: base vs distilled

A small blind comparison of the base model and the distilled model on the evaluation prompts. It exists
because the automatic proxy metrics (right script, looping, length) cannot judge whether an answer is
correct.

## Result

40 prompts (10 per task), one rater, scores from 1 to 5. Mean scores, base -> distilled:

| task | n | fluency | correctness | distilled wins / ties / base wins (correctness) |
|---|---|---|---|---|
| closed-book knowledge | 10 | 3.30 -> 3.90 | 3.60 -> 4.10 | 4 / 4 / 2 |
| instruction following | 10 | 3.60 -> 3.40 | 4.00 -> 3.80 | 4 / 3 / 3 |
| summarization | 10 | 3.50 -> 3.70 | 3.70 -> 3.40 | 3 / 2 / 5 |
| translation | 10 | 4.00 -> 3.80 | 3.30 -> 4.00 | 6 / 1 / 3 |
| **overall** | 40 | 3.60 -> 3.70 | 3.65 -> 3.83 | 17 / 10 / 13 |

Reading: the human ratings do not show a clear difference between the two models. Overall correctness is
+0.18 and fluency +0.10 for the distilled model. Excluding ties, the overall split of 17 to 13 gives an exact
two-sided sign-test p-value of 0.585, and no single task reaches significance (translation 6 to 3, p = 0.51;
summarization 3 to 5, p = 0.73). The gains seen in the automatic metrics (right script, fewer runaway answers)
did not turn into a measurable gain in rated content quality.

## Limits
- One rater, who is also the author of the project.
- 10 prompts per task, one random draw (seed 11).
- Blinding is imperfect: answers may differ in style and length in ways that reveal the model.
- Correctness was judged from the rater's own knowledge; no reference answers were used.
- Ratings were made against the full prompt and both answers.

## Rubric
- Fluency (1-5): 1 garbled or wrong language, 3 understandable with clear errors, 5 natural.
- Correctness (1-5): 1 wrong, unrelated or empty, 3 partly right, 5 fully right.
  - Knowledge: facts must be right; if a claim could not be verified, no more than 3.
  - Instruction following: answers the question asked, correctly, from the passage.
  - Translation: meaning preserved in the target language.
  - Summarization: faithful to the passage, main point covered, nothing invented.

## Files
| file | what it is |
|---|---|
| `make_items.py` | Samples 10 prompts per task, randomizes which model is answer A or B, writes `rating_items.json` and `rating_key.csv`. |
| `rating_items.json` | The 40 blind items: task, prompt, answer A, answer B. |
| `rating_key.csv` | Which model wrote answer A and answer B for each item. Not shown to the rater. |
| `rate_app.py` | Streamlit app for rating. Writes `ratings.csv` after each completed item. Never reads the key. |
| `ratings.csv` | The ratings: `eval_id`, `fluency_A`, `correct_A`, `fluency_B`, `correct_B`. |
| `score.py` | Joins ratings with the key, prints the table above, writes `rating_summary.csv`. |
| `rating_summary.csv` | The summary table in CSV form. |

## Reproduce the table
No installation is needed (standard library only):

    python3 human_eval/score.py

## Rate again from scratch
`make_items.py` overwrites `rating_items.json` and `rating_key.csv`, which makes the committed
`ratings.csv` invalid. Work in a copy of the folder so the original ratings are kept:

    cp -r human_eval human_eval_new && cd human_eval_new
    rm ratings.csv rating_summary.csv
    # optional: change the random seed in make_items.py to draw different prompts
    python3 make_items.py
    pip install -U streamlit
    streamlit run rate_app.py --server.address localhost
    python3 score.py

`make_items.py` reads `eval_prompts_60.csv` and `results/simachew.csv` from the repository root, so run the
copy from inside the repository (as above) or copy those two files alongside it. Do not open
`rating_key.csv` until the rating is finished.

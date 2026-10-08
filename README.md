# Amharic response-level distillation (Gemma-3-270M-it + LoRA)

Response-level distillation of Amharic ability into `google/gemma-3-270m-it`. The student is fine-tuned
with LoRA on teacher responses, then compared with the untouched base model on 60 evaluation prompts, by
automatic proxy metrics and by a blind human rating.

## Pipeline
1. **Prompts and teacher answers** (`generate_data.py`, output in `data/`): 2,612 raw synthetic prompts were
   filtered to 2,000 (script mix, duplicates, near-duplicates of the eval prompts, passage length, and others;
   counts in `data/prompts_clean.meta.json`). The teacher `gemma-4-26b-a4b-it` (Gemini API) answered them;
   after filtering, 1,934 pairs remain (`data/train_gemini_gemma-4-26b-a4b-it.csv`).
   Rows by task: closed-book knowledge 451, instruction following 500, translation 487, summarization 496.
   Prompts for 1,833 rows were written by the same model that answered them; 101 rows have prompts from
   Gemini Flash models. Every row is labelled synthetic in its `source` column.
2. **Training and evaluation** (`notebooks/amharic_distillation_colab.ipynb`, Colab T4): base run, LoRA training,
   distilled run, comparison. A leak check found 0 training prompts overlapping the 60 evaluation prompts.
3. **Human evaluation** (`human_eval/`): a blind comparison of base and distilled answers, 10 prompts per task.

## Setup
- Student: `google/gemma-3-270m-it`; LoRA r=16, alpha=32, dropout 0.05 on all attention and MLP projections
  (3,796,992 trainable parameters); 3 epochs, 345 steps, effective batch 16, lr 2e-4 with cosine schedule,
  fp32, seed 42; loss on answer tokens only; 1,834 train / 100 validation pairs.
- Evaluation: greedy decoding, 256 new tokens, batch size 1, Tesla T4.
- Evaluation prompts: `eval_prompts_60.csv`, 15 per task, provided by the task organizers.

## Results

Validation loss on held-out teacher answers (same synthetic pool as training, so in-distribution):
3.738 (before) -> 1.631 -> 1.518 -> **1.506** (epoch 3).

Efficiency (Tesla T4):

| metric | base | distilled (adapter loaded) | distilled (merged) |
|---|---|---|---|
| parameters (total) | 268,098,176 | 271,895,168 | 268,098,176 |
| trainable parameters (LoRA) | 0 | 3,796,992 | 0 |
| adapter size on disk (MB) | - | 14.52 | - |
| peak inference memory (MB) | 1158 | 1181 | 1158 |
| tokens per second | 25.7 | 21.1 | 32.8 |
| seconds per answer (mean) | 3.99 | 3.35 | 2.15 |
| answers that hit 256 tokens | 17 | 1 | 1 |

Wall-clock timings on shared Colab GPUs vary between sessions, so compare tokens per second and treat small
differences as noise. Mean seconds per answer are not comparable across models because the outputs differ in length.

Proxy metrics on the 60 evaluation prompts (15 per task; no reference answers):

| task | right script % (base -> distilled) | looping % | hit 256 tokens % |
|---|---|---|---|
| closed_book_knowledge | 26.7 -> 93.3 | 33.3 -> 20.0 | 33.3 -> 6.7 |
| instruction_following | 33.3 -> 100 | 13.3 -> 0 | 26.7 -> 0 |
| translation | 6.7 -> 100 | 0 -> 0 | 0 -> 0 |
| summarization | 53.3 -> 100 | 6.7 -> 0 | 53.3 -> 0 |

Blind human rating (40 prompts, 10 per task, one rater, scores 1 to 5; details in `human_eval/README.md`):

| | base | distilled |
|---|---|---|
| mean fluency | 3.60 | 3.70 |
| mean correctness | 3.65 | 3.83 |
| head to head on correctness (distilled wins / ties / base wins) | | 17 / 10 / 13 |

Excluding ties, the 17 to 13 split gives an exact two-sided sign-test p-value of 0.585, so the human ratings
do not show a clear difference between the models. No single task reaches significance.

Checkpoint comparison (chrF++ against the teacher on 60 held-out validation prompts, in-distribution):

| checkpoint | val loss | knowledge | instruction following | translation | summarization |
|---|---|---|---|---|---|
| epoch 1 | 1.631 | 22.0 | 48.3 | 15.4 | 35.9 |
| epoch 2 | 1.518 | 22.1 | 44.1 | 22.2 | 34.4 |
| epoch 3 | 1.506 | 22.9 | 42.1 | 23.1 | 31.9 |

The final adapter is epoch 3. On the evaluation prompts it loops and hits the length limit least in closed-book
answers (20% loops vs 33% and 40%; 7% hit 256 tokens vs 33% and 33%). Instruction-following and summarization
chrF++ were highest after epoch 1.

## What the fine-tuning fixes
The base model drifts into other scripts (Tamil, Gujarati, Bengali) and often does not stop. After LoRA the model
answers in the right script and ends its answers. The blind human rating shows no clear gain in content quality.

## Limitations
- **No measurable gain in rated content quality.** "Right script" measures the writing system, not correctness.
  The blind rating gave similar scores to both models (correctness 3.65 vs 3.83, not significant). Failure
  modes seen in the evaluation outputs:
  - E01: degenerate repetition and a wrong answer.
  - E02: restates the question, then drifts to something unrelated.
  - E16, E17: answers a different question from the one asked.
  - E32: the translation is fluent but wrong.
  - E47: the summary is one sentence lifted from the prompt.
- **Training prompts come from one prompt-writing model, mostly the teacher itself** (1,833 of 1,934 rows).
  Training and validation therefore sample the teacher's own prompt distribution, so validation loss and chrF++
  against the teacher are in-distribution and optimistic about transfer. The 60 evaluation prompts were written
  independently by the task organizers, so the base-versus-distilled comparison does not rest on teacher-written
  prompts, but generalization to prompts from other sources is not shown.
- **Training and evaluation prompts differ in wording and format.** Instruction-following training prompts use
  only 11 distinct five-word openings across 500 prompts (summarization 83 across 496, translation 45 across 487,
  knowledge 311 across 451), and none of the 15 evaluation prompts in any task starts with a training opening.
  Three of the 15 evaluation summarization prompts end in a scraped headline line; none of the 1,934 training
  prompts does. Prompt lengths are similar (median words, train vs eval: knowledge 11 vs 12, instruction
  following 111 vs 112, summarization 144 vs 173, translation 19 vs 21).
- **Closed-book looping remains.** A repeated-trigram heuristic flags 20% of closed-book answers (3 of 15).
  The heuristic probably undercounts: a looser repetition check flags most long closed-book answers.
- **Translation and knowledge are the weakest tasks** by validation loss (translation 2.58, knowledge 2.17,
  summarization 1.05, instruction following 0.35 at epoch 3).
- **Heavy copying from the prompt.** In instruction-following and summarization answers, 95% and 83% of
  character 4-grams also occur in the prompt (epoch 3). These metrics cannot tell correct extraction from
  copying the wrong sentence.
- **Overfitting after epoch 2.** Mean training loss falls from about 1.44 (epoch 2) to 1.25 (epoch 3) while
  validation loss stays flat (1.518 -> 1.506), so more epochs on this data are not expected to help.
- **Small evaluations.** 15 prompts per task and one seed for the proxy metrics; one rater, who is also the
  author, and 10 prompts per task for the human rating, with imperfect blinding.
- **No reference-based score on the 60 evaluation prompts.** chrF++ against the teacher was computed only on
  held-out validation prompts, and not for the base model.
- **Claim supported by the evidence:** improved script purity and fewer runaway answers, with no measurable
  change in human-rated content quality.

## Reproduce
1. Open the notebook in Colab with a T4 GPU and add your Hugging Face token as the secret `HF_TOKEN`.
2. Put `data/train_gemini_gemma-4-26b-a4b-it.csv`, `eval_prompts_60.csv` and `predictions_template.csv` in
   `MyDrive/amharic_distill_out/`. `predictions_template.csv` is not included in this repository.
3. Run the cells top to bottom. Training takes about 26 minutes on a T4.
4. The human-evaluation table can be reproduced without any installation: `python3 human_eval/score.py`.

## Files
- `generate_data.py`, `data/`: data generation pipeline and its outputs (`requirements.txt`)
- `notebooks/`: the Colab notebook with saved outputs (`requirements_colab.txt`)
- `results/`: predictions, efficiency table, per-task comparison, checkpoint comparison, training logs
- `human_eval/`: blind human evaluation (items, key, ratings, scripts; see its README)
- `adapter/`: LoRA weights (epoch 3)
- `eval_prompts_60.csv`: the 60 evaluation prompts, provided by the task organizers

## Licenses
The MIT license (`LICENSE`) covers the code in this repository. Other parts follow their own terms:
- The adapter is built on `google/gemma-3-270m-it`, which is distributed under the Gemma Terms of Use; read them
  before reusing the adapter.
- The training data is model-generated text. The teacher `gemma-4-26b-a4b-it` is released under Apache 2.0;
  101 prompts were written by Gemini Flash models through the Gemini API, which has its own terms.
- The evaluation prompts were provided by the task organizers, and their terms apply.

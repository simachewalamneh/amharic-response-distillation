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

## Student size: Gemma-3-1B-it (same recipe)

An additional experiment with a larger student, to test whether the weak content quality is a capacity limit.
It uses the same 1,934 teacher pairs, seed, LoRA settings (r=16, alpha=32), 3 epochs and evaluation as the main
run; only the student changes (`google/gemma-3-1b-it`, 999,885,952 parameters). Results are in `results_1b/`.
The 1B adapter is not included in this repository.

Changes made by hand in Colab to fit the T4, besides the model name: micro-batch 1 with gradient accumulation 16
(effective batch 16, as before); validation in batches of 1 instead of 4 to avoid an out-of-memory error (the
loss is token-weighted, so the values agree up to floating-point noise); and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Distilled models, 270M vs 1B:

| | 270M | 1B |
|---|---|---|
| validation loss, epoch 3 | 1.506 | 1.161 |
| best validation loss | 1.506 (epoch 3) | 1.149 (epoch 2) |
| training time | 25.6 min | 68.3 min |
| peak training memory | 9626 MB (micro-batch 2) | 7191 MB (micro-batch 1) |
| adapter size | 14.52 MB | 49.82 MB |
| full model on disk | 511 MB | 1907 MB |
| peak inference memory, merged | 1158 MB | 4191 MB |
| tokens per second, merged | 32.8 | 21.4 |

Proxy metrics on the 60 evaluation prompts, distilled models (right script % / looping % / hit 256 tokens %):

| task | 270M | 1B |
|---|---|---|
| closed_book_knowledge | 93.3 / 20.0 / 6.7 | 100 / 20.0 / 26.7 |
| instruction_following | 100 / 0 / 0 | 100 / 0 / 0 |
| translation | 100 / 0 / 0 | 93.3 / 0 / 0 |
| summarization | 100 / 0 / 0 | 100 / 0 / 0 |

Reading:
- Validation loss on held-out teacher answers is much lower for the 1B student. It is measured on
  teacher-written prompts, so it is in-distribution, and it is comparable across the two models only if they
  share a tokenizer (not verified here).
- The proxy metrics do not separate the two students: every difference is one answer (6.7 points). Closed-book
  answers are longer for 1B (mean 140.5 vs 101.3 new tokens), so more of them reach the 256-token limit.
  These proxies were already saturated at 270M.
- Whether the lower loss means better answers has not been tested: no blind human comparison of the two students
  has been done.
- The 1B base model is more verbose and drifts more than the 270M base (closed-book right script 6.7% vs 26.7%;
  32 of 60 answers hit 256 tokens vs 17), so the effect of fine-tuning looks larger for 1B partly because its
  base starts lower.
- Timings come from different Colab sessions and are only approximate.


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
- `results_1b/`: results of the 1B student experiment (predictions, efficiency table, training logs)
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

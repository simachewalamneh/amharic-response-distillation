# Amharic response-level distillation (Gemma-3-270M-it + LoRA)

Response-level distillation of Amharic ability into `google/gemma-3-270m-it`. The student is fine-tuned
with LoRA on teacher responses, then compared with the untouched base model on 60 evaluation prompts.

## Pipeline
1. **Prompts and teacher answers** (`generate_data.py`, output in `data/`): 2,612 raw synthetic prompts were
   filtered to 2,000 (script mix, duplicates, near-duplicates of the eval prompts, passage length, and others;
   counts in `data/prompts_clean.meta.json`). The teacher `gemma-4-26b-a4b-it` (Gemini API) answered them;
   after filtering, 1,934 pairs remain (`data/train_gemini_gemma-4-26b-a4b-it.csv`).
   Rows by task: closed-book knowledge 451, instruction following 500, translation 487, summarization 496.
   Prompts for 1,833 rows came from the same teacher model; 101 rows have prompts from Gemini Flash models.
   Every row is labelled synthetic in its `source` column.
2. **Training and evaluation** (`notebooks/amharic_distillation_colab.ipynb`, Colab T4): base run, LoRA training,
   distilled run, comparison. A leak check found 0 training prompts overlapping the 60 evaluation prompts.

## Setup
- Student: `google/gemma-3-270m-it`; LoRA r=16, alpha=32, dropout 0.05 on all attention and MLP projections
  (3,796,992 trainable parameters); 3 epochs, 345 steps, effective batch 16, lr 2e-4 with cosine schedule,
  fp32, seed 42; loss on answer tokens only; 1,834 train / 100 validation pairs.
- Evaluation: greedy decoding, 256 new tokens, batch size 1, Tesla T4.

## Results

Validation loss on held-out teacher answers: 3.738 (before) -> 1.631 -> 1.518 -> **1.506** (epoch 3).

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

Checkpoint comparison (chrF++ against the teacher on 60 held-out validation prompts, about 15 per task):

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
answers in the right script and ends its answers.

## Limitations
- "Right script" measures the writing system, not correctness. Many distilled answers are fluent but wrong or
  unfaithful; see `results/simachew.csv` and `results/distilled_outputs.json`.
- Closed-book answers still loop in some cases, and translation has the highest validation loss.
- Instruction-following and summarization outputs copy heavily from the prompt (83-97% of character 4-grams
  also occur in the prompt), which these metrics cannot tell apart from correct extraction.
- Training loss keeps falling in epoch 3 (about 1.25) while validation loss is flat (1.506 vs 1.518 at epoch 2),
  so more epochs on this data are not expected to help.
- Only 15 prompts per task and one seed: each prompt moves a percentage by 6.7 points.
- No reference-based score against the teacher on the 60 evaluation prompts.

## Reproduce
1. Open the notebook in Colab with a T4 GPU and add your Hugging Face token as the secret `HF_TOKEN`.
2. Put `train_gemini_gemma-4-26b-a4b-it.csv`, `eval_prompts_60.csv` and `predictions_template.csv` in
   `MyDrive/amharic_distill_out/`. The evaluation prompts are not included in this repository.
3. Run the cells top to bottom. Training takes about 26 minutes on a T4.

## Files
- `generate_data.py`, `data/`: data generation pipeline and its outputs (`requirements.txt`)
- `notebooks/`: the Colab notebook with saved outputs (`requirements_colab.txt`)
- `results/`: predictions, efficiency table, per-task comparison, checkpoint comparison, training logs
- `adapter/`: LoRA weights (epoch 3)

## Licenses
The adapter is built on `google/gemma-3-270m-it`, which is distributed under the Gemma Terms of Use; read them
before reusing it. The teacher `gemma-4-26b-a4b-it` is released under Apache 2.0.

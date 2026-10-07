GPU: **Tesla T4** | dtype torch.float32 | greedy decoding, max_new_tokens=256, batch size 1 | seed 42

| metric | base | distilled (adapter loaded) | distilled (merged) |
|---|---|---|---|
| parameters (total) | 268,098,176 | 271,895,168 | 268,098,176 |
| trainable parameters (LoRA) | 0 | 3,796,992 | 0 |
| size on disk, full model (MB) | 511 | 511 | 511 |
| adapter size on disk (MB) | - | 14.52 | - |
| peak inference memory (MB) | 1158 | 1181 | 1158 |
| seconds per answer (mean) | 3.99 | 3.35 | 2.15 |
| seconds per answer (median) | 2.02 | 2.18 | 1.69 |
| tokens per second | 25.7 | 21.1 | 32.8 |
| answers that hit 256 tokens | 17 | 1 | 1 |

Training: 25.6 min on Tesla T4 | 3 epochs, 345 steps, effective batch 16, lr 0.0002, LoRA r=16 alpha=32 dropout=0.05, max length 1024 | peak training memory 9626 MB | 1834 train / 100 validation pairs
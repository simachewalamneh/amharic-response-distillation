GPU: **Tesla T4** | dtype torch.float32 | greedy decoding, max_new_tokens=256, batch size 1 | seed 42

| metric | base | distilled (adapter loaded) | distilled (merged) |
|---|---|---|---|
| parameters (total) | 999,885,952 | 1,012,931,712 | 999,885,952 |
| trainable parameters (LoRA) | 0 | 13,045,760 | 0 |
| size on disk, full model (MB) | 1907 | 1907 | 1907 |
| adapter size on disk (MB) | - | 49.82 | - |
| peak inference memory (MB) | 4034 | 4272 | 4191 |
| seconds per answer (mean) | 8.23 | 6.36 | 3.61 |
| seconds per answer (median) | 11.09 | 4.62 | 2.75 |
| tokens per second | 21.8 | 12.2 | 21.4 |
| answers that hit 256 tokens | 32 | 4 | 4 |

Training: 68.3 min on Tesla T4 | 3 epochs, 345 steps, effective batch 16, lr 0.0002, LoRA r=16 alpha=32 dropout=0.05, max length 1024 | peak training memory 7191 MB | 1834 train / 100 validation pairs
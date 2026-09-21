# Smoke results — ai-learn-02-positional-encoding

**Task:** marker-quarter classification (which of 4 sequence quarters holds token id=0).

**Seed:** `42`

## Headline metrics

| Condition | Train acc | Test acc | Marker select |
|-----------|----------:|---------:|--------------:|
| No PE | 0.3271 | 0.2344 | 0.0000 |
| Sinusoidal PE | 1.0000 | 1.0000 | 1.0000 |
| Learned PE | 1.0000 | 1.0000 | 1.0000 |
| Chance | — | 0.2500 | — |

| Metric | Value |
|--------|------:|
| Runtime (s) | 2.08 |
| Unit checks | PASS |

## Config

- `seq_len=16`, `vocab=20`, `d_model=32`, 4 quarter classes
- train `1024` / test `256`, batch `64`, epochs `80`, lr `0.2`
- model: content-gated readout — a=softmax(X w); **no PE** uses h=a·X; **with PE** uses h=a·PE

## Plots

- [`pe_heatmap.png`](pe_heatmap.png) / [`pe_heatmap.svg`](pe_heatmap.svg) — sinusoidal PE matrix (dims × positions)
- [`position_similarity.png`](position_similarity.png) / [`position_similarity.svg`](position_similarity.svg) — cosine similarity between PE rows
- [`accuracy_comparison.png`](accuracy_comparison.png) / [`accuracy_comparison.svg`](accuracy_comparison.svg) — no PE vs sinusoidal vs learned

## Takeaway

Content attention can **find** the marker without PE, but the soft-selected content vector is nearly the same regardless of *where* it sat — so the no-PE probe stays near chance (~0.25).
Reading out **PE at the selected position** (sinusoidal or learned) supplies absolute location, and the linear head recovers the quarter.

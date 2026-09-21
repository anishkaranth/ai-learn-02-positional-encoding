# AI Learn 02 — Sinusoidal Positional Encoding from Scratch

Second project in the **ai-learn-*** series (after ai-learn-01 attention). Transformers have no recurrence and no convolution — without **positional encoding**, a bag of token embeddings cannot tell *where* each token sat.

Implement the Vaswani et al. **sinusoidal** PE and an optional **learned** PE table in plain **NumPy**, visualize the geometry, and prove PE matters on a marker-quarter classification smoke task.

## Learning goals

- Why self-attention is **permutation-equivariant** without position signals
- The **sinusoidal** formula: `PE(pos, 2i) = sin(pos / 10000^(2i/d))`, `PE(pos, 2i+1) = cos(...)`
- How a **learned** position embedding table `(max_len, d_model)` compares (BERT-style)
- Heatmap of encodings, nearby-position **cosine similarity**, and adding PE to toy token embeddings
- Unit-style checks (shape, range/finite, dim0/dim1 = sin/cos, nearby ≻ far)
- Empirically: content-gated readout **without PE ≈ chance**; with sinusoidal or learned PE → ~1.0 accuracy

## Project layout

```
README.md
requirements.txt
positional_encoding.py
pe_smoke_core.py / pe_smoke_plots.py
run_smoke.py
notebooks/positional_encoding.ipynb
results/
  RESULTS.md
  metrics.json
  JSON.shot
  pe_heatmap.svg / position_similarity.svg / accuracy_comparison.svg
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_smoke.py
```

Smoke finishes in well under ~30 seconds on CPU, prints test accuracy for no-PE / sinusoidal / learned, and refreshes `results/`.

**Notebook:**

```bash
jupyter notebook notebooks/positional_encoding.ipynb
```

## Core API

```python
from positional_encoding import (
    sinusoidal_positional_encoding,
    LearnedPositionalEncoding,
    add_positional_encoding,
    position_cosine_similarity,
    check_sinusoidal_properties,
)

pe = sinusoidal_positional_encoding(seq_len=16, d_model=32)  # (T, D)
X_pe = add_positional_encoding(X, pe)

learned = LearnedPositionalEncoding(max_len=16, d_model=32)
X_pe = add_positional_encoding(X, learned.forward())
```

## Math (sinusoidal)

\[
\mathrm{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right), \quad
\mathrm{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right)
\]

Nearby positions have similar PE vectors (high cosine similarity); far positions diverge — see `results/position_similarity.svg`.

## Smoke task (optional demo)

Sequences of length `T = 16`, vocab `V = 20`:

1. Random tokens in `1 … V-1`, exactly one **marker** token `id = 0`
2. Label = which of **4 quarters** holds the marker (chance = 0.25)
3. Content-gated readout: `α = softmax(X w)`; no PE uses `h = α·X`; with PE uses `h = α·PE`

## Dependencies

Pinned lightly in `requirements.txt`: **numpy**, **matplotlib**, **jupyter**. CPU-only.

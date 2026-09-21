#!/usr/bin/env python3
"""Smoke demo: marker-quarter classification — prove positional encoding matters."""
from pe_smoke_core import *  # noqa: F403

def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)

    print("Running unit checks...")
    unit = run_unit_checks()
    for k, v in unit.items():
        if k != "all_passed":
            print(f"  {k}: {'PASS' if v else 'FAIL'}")
    assert unit["all_passed"], f"Unit checks failed: {unit}"
    print("Unit checks: PASS\n")

    tokens_tr, y_tr, _ = make_batch(rng, N_TRAIN)
    tokens_te, y_te, _ = make_batch(rng, N_TEST)
    print(
        "Train label counts:",
        {int(c): int(n) for c, n in zip(*np.unique(y_tr, return_counts=True))},
    )

    t0 = time.time()
    results = {}
    for mode in ("none", "sinusoidal", "learned"):
        print(f"\nTraining condition: {mode}")
        seed_off = {"none": 0, "sinusoidal": 1, "learned": 2}[mode]
        cond_rng = np.random.default_rng(SEED + seed_off)
        results[mode] = train_condition(mode, tokens_tr, y_tr, tokens_te, y_te, cond_rng)
    elapsed = time.time() - t0

    pe = sinusoidal_positional_encoding(SEQ_LEN, D_MODEL)
    plot_pe_heatmap(pe, RESULTS / "pe_heatmap.png")
    plot_position_similarity(pe, RESULTS / "position_similarity.png")
    plot_accuracy_comparison(results, RESULTS / "accuracy_comparison.png")

    metrics = {
        "project": "ai-learn-02-positional-encoding",
        "task": "marker_quarter_classification",
        "random_seed": SEED,
        "config": {
            "seq_len": SEQ_LEN,
            "vocab": VOCAB,
            "marker_id": MARKER,
            "n_classes": N_CLASSES,
            "d_model": D_MODEL,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "batch_size": BATCH,
            "epochs": EPOCHS,
            "lr": LR,
            "learned_pe_lr": PE_LR,
            "model": "content_gated_PE_readout (a=softmax(Xw); h=aX or a@PE)",
        },
        "metrics": {
            "no_pe_train_accuracy": results["none"]["final_train_acc"],
            "no_pe_test_accuracy": results["none"]["final_test_acc"],
            "sinusoidal_train_accuracy": results["sinusoidal"]["final_train_acc"],
            "sinusoidal_test_accuracy": results["sinusoidal"]["final_test_acc"],
            "learned_train_accuracy": results["learned"]["final_train_acc"],
            "learned_test_accuracy": results["learned"]["final_test_acc"],
            "no_pe_marker_selection_accuracy": results["none"]["marker_selection_accuracy"],
            "sinusoidal_marker_selection_accuracy": results["sinusoidal"]["marker_selection_accuracy"],
            "learned_marker_selection_accuracy": results["learned"]["marker_selection_accuracy"],
            "chance_accuracy": 0.25,
            "unit_checks_passed": True,
            "unit_checks": {k: bool(v) for k, v in unit.items()},
        },
        "history": {
            mode: results[mode]["history"] for mode in ("none", "sinusoidal", "learned")
        },
        "runtime_seconds": round(elapsed, 3),
        "pe_shape": list(pe.shape),
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    shot = {
        "snapshot": "smoke_run",
        "seed": SEED,
        "no_pe_test_accuracy": results["none"]["final_test_acc"],
        "sinusoidal_test_accuracy": results["sinusoidal"]["final_test_acc"],
        "learned_test_accuracy": results["learned"]["final_test_acc"],
        "chance_accuracy": 0.25,
        "runtime_seconds": round(elapsed, 3),
        "unit_checks_passed": True,
        "plots": [
            "pe_heatmap.png",
            "position_similarity.png",
            "accuracy_comparison.png",
            "pe_heatmap.svg",
            "position_similarity.svg",
            "accuracy_comparison.svg",
        ],
    }
    (RESULTS / "JSON.shot").write_text(json.dumps(shot, indent=2) + "\n")

    results_md = f"""# Smoke results — ai-learn-02-positional-encoding

**Task:** marker-quarter classification (which of 4 sequence quarters holds token id=0).

**Seed:** `{SEED}`

## Headline metrics

| Condition | Train acc | Test acc | Marker select |
|-----------|----------:|---------:|--------------:|
| No PE | {results['none']['final_train_acc']:.4f} | {results['none']['final_test_acc']:.4f} | {results['none']['marker_selection_accuracy']:.4f} |
| Sinusoidal PE | {results['sinusoidal']['final_train_acc']:.4f} | {results['sinusoidal']['final_test_acc']:.4f} | {results['sinusoidal']['marker_selection_accuracy']:.4f} |
| Learned PE | {results['learned']['final_train_acc']:.4f} | {results['learned']['final_test_acc']:.4f} | {results['learned']['marker_selection_accuracy']:.4f} |
| Chance | — | 0.2500 | — |

| Metric | Value |
|--------|------:|
| Runtime (s) | {elapsed:.2f} |
| Unit checks | PASS |

## Config

- `seq_len={SEQ_LEN}`, `vocab={VOCAB}`, `d_model={D_MODEL}`, 4 quarter classes
- train `{N_TRAIN}` / test `{N_TEST}`, batch `{BATCH}`, epochs `{EPOCHS}`, lr `{LR}`
- model: content-gated readout — a=softmax(X w); **no PE** uses h=a·X; **with PE** uses h=a·PE

## Plots

- [`pe_heatmap.png`](pe_heatmap.png) / [`pe_heatmap.svg`](pe_heatmap.svg) — sinusoidal PE matrix (dims × positions)
- [`position_similarity.png`](position_similarity.png) / [`position_similarity.svg`](position_similarity.svg) — cosine similarity between PE rows
- [`accuracy_comparison.png`](accuracy_comparison.png) / [`accuracy_comparison.svg`](accuracy_comparison.svg) — no PE vs sinusoidal vs learned

## Takeaway

Content attention can **find** the marker without PE, but the soft-selected content vector is nearly the same regardless of *where* it sat — so the no-PE probe stays near chance (~0.25).
Reading out **PE at the selected position** (sinusoidal or learned) supplies absolute location, and the linear head recovers the quarter.
"""
    (RESULTS / "RESULTS.md").write_text(results_md)

    print("\n=== Headline ===")
    print(
        f"no PE        test_acc={results['none']['final_test_acc']:.4f}  "
        f"sel={results['none']['marker_selection_accuracy']:.3f}"
    )
    print(
        f"sinusoidal   test_acc={results['sinusoidal']['final_test_acc']:.4f}  "
        f"sel={results['sinusoidal']['marker_selection_accuracy']:.3f}"
    )
    print(
        f"learned      test_acc={results['learned']['final_test_acc']:.4f}  "
        f"sel={results['learned']['marker_selection_accuracy']:.3f}"
    )
    print(f"runtime      {elapsed:.2f}s")
    print(f"wrote {RESULTS}")

    assert results["sinusoidal"]["final_test_acc"] > 0.7, "sinusoidal PE should crush chance"
    assert results["learned"]["final_test_acc"] > 0.7, "learned PE should crush chance"
    assert results["none"]["final_test_acc"] < 0.45, "no-PE should stay near chance"


if __name__ == "__main__":
    main()

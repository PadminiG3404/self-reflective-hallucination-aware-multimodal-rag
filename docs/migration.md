# Migration Notes

## 2026-05-24 Enhancements

- Hallucination reporting now includes factor diagnostics, dominant factor, and severity levels.
- Optional reflection reports can be attached to final explanations.
- Uncertainty reports include optional ECE and miscalibration scores when provided.
- New evaluation features include failure analysis export and optional Weights & Biases tracking.
- New YAML configuration keys were added under `reasoning`, `hallucination`, `uncertainty`, `reflection`, and `evaluation`.

## Compatibility

- Existing pipeline interfaces remain compatible. New fields are optional and default safely.
- If ablation flags are not provided, the pipeline behaves as before.

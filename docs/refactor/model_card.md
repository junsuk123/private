# Strategy Utility Model Card

Status: architecture implemented; CPU/OpenVINO shadow only; not promoted for trading.

The model is a fixed-shape dense temporal R-GCN implemented in `app.models.strategy_utility`. Topology construction remains outside the model. Runtime graph operations are fixed Gather, MatMul, Add, ReLU, Multiply, Concat, and Squeeze; there are no dynamic sparse/scatter operations.

Outputs are per stock-strategy success, gross return, cost, MAE, MFE, fill probability, holding time, uncertainty, utility, and NoTrade probability. Hard ontology masks are applied outside the learned graph and cannot be overridden.

Current weights are deterministic initialization for runtime/parity verification,
not a trained production checkpoint. The stored-data evaluation produced 34,860
causal labels and a purged/embargoed tabular baseline, but its 11-date,
US-concentrated coverage lacks the point-in-time event, sector, session, and
legacy-decision fields required to train or compare the full model honestly.
Consequently no claim of out-of-sample alpha, calibration, or superiority over
tabular baselines is made. Promotion still requires representative KRX/NXT
coverage, calibration curves, regime/symbol results, block-bootstrap confidence
intervals, and multiple-testing correction.

Model hash benchmarked: `6acf71cd0718e3738b9725df0b616f5cf9bf21892b3592b7952340b050a99ed4`.

# CVAE Beta Selection Plan

## Goal

Select a disease-specific beta for the EDDI conditional VAE. A single global beta is unlikely to be optimal because each disease has different target indicators, prevalence, threshold rules, missingness patterns, and source predictability.

## Final Experiment Grid

- Model: base EDDI CVAE
- Dataset: `harmonized_knhanes_1998_2024`
- Betas: `0.001`, `0.01`, `0.1`, `1.0`
- Repeats: `5` per beta per disease
- Diseases:
  - `diabetes`
  - `hypertension`
  - `dyslipidemia`
  - `liver_disease`
  - `hepatitis_b`
  - `hepatitis_c`
  - `kidney_disease`
  - `anemia`
- Training:
  - `steps: 2000`
  - `beta_warmup_steps: 1000`
  - early stopping enabled
  - otherwise follow the base CVAE training config

## Evaluation Metrics

Evaluate every trained model with:

1. Collapse / failed reconstruction
2. Coverage calibration
3. Conditional prior reconstruction
4. AUROC

### Collapse / Failed Reconstruction

Use the existing best-valid checkpoint metrics.

```text
classic KL collapse:
  best_valid_kl < 0.1

failed reconstruction:
  best_valid_recon > 0.1

unusable CVAE run:
  best_valid_kl < 0.1 OR best_valid_recon > 0.1
```

Collapse/failed runs should be reported, but excluded from beta quality comparisons unless explicitly analyzing failure rate.

### Coverage Calibration

Coverage should be evaluated from conditional prior samples.

Primary levels:

```text
0.90
0.95
0.99
```

Report at least:

```text
coverage
absolute coverage error
mean interval width
median interval width
mean bias
absolute mean bias
```

Important: interval width alone is not sufficient. A smaller width can still have better coverage if the generated distribution is better centered. Always interpret width together with bias.

### Conditional Prior Reconstruction

Use the mean of multiple conditional prior samples as the point estimate.

Report:

```text
transformed MAE / RMSE
raw MAE / RMSE by target feature
overall raw MAE only as a rough reference
```

Raw overall MAE can mix different units, so feature-level raw MAE and transformed-space MAE are more reliable for beta comparison.

### AUROC

AUROC is disease-threshold/ranking performance. It can disagree with reconstruction and coverage because a model may reconstruct average values well but rank positive cases poorly.

Report:

```text
mean AUROC over usable runs
max AUROC over usable runs
usable run count
```

## Disease-Specific Selection Rule

For each disease, choose beta in this order:

1. Exclude unusable runs from performance comparison.
2. Prefer beta values with acceptable usable-run rate.
3. Among usable runs, compare coverage calibration.
4. Compare prior reconstruction.
5. Compare AUROC.
6. Choose based on the disease's intended use:
   - Screening or prediction: prioritize AUROC and sensitivity.
   - Generative imputation: prioritize coverage calibration and prior reconstruction.
   - Uncertainty analysis: prioritize coverage error, interval width, and bias.

## Recommended Result Tables

Per disease:

```text
disease | beta | usable/5 | AUROC mean | AUROC max | coverage90 error | width90 | prior transformed MAE | feature raw MAE
```

Final selection:

```text
disease | selected beta | reason | caveat
```

## Current Lessons From Diabetes Experiments

Diabetes showed that:

- `beta=0.001` is stable but prior reconstruction and AUROC can be weak.
- `beta=0.01` is strong when it trains well, but can still collapse.
- Higher beta can improve sharpness, prior reconstruction, and AUROC among usable runs.
- Very high beta values, especially `0.5` and `1.0`, collapse too often with the current warmup setup.
- `beta_warmup_steps=1000` helps, but does not fully remove collapse.

Therefore the final grid keeps both low and high beta values:

```text
0.001: stable low-beta reference
0.01: strong practical candidate
0.1: high-beta performance probe
1.0: extreme beta stress test
```


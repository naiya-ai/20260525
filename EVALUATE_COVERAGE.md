# CVAE Coverage Evaluation

This document defines how to evaluate coverage for CVAE-generated target
indicator distributions.

## Goal

Measure whether the CVAE conditional prior produces target distributions that
contain the observed target values at the expected rate.

For each source row, the model samples targets from `p(target | source)`.
Coverage asks whether the observed target value falls inside the generated
prediction interval.

## Feature Selection

Coverage evaluation must not hard-code disease-specific feature names.

By default, evaluate every numerical target feature listed in the target
group's `metadata.json`:

```text
metadata["features"]["num"]
```

For example, the diabetes target currently contains numerical features such as
glucose and HbA1c, but the evaluator should discover those from metadata rather
than naming them in the protocol.

Categorical targets are excluded from the default coverage metric because they
need a separate discrete calibration definition. A categorical extension can be
added later using probability mass assigned to the observed category or set
coverage over high-probability categories.

## Protocol

Evaluation should run for every configured disease/target group, not for a
single disease only. For each CVAE checkpoint, target group, split, and
numerical target feature:

1. Load source rows and observed target values from the configured split.
2. For each source row, sample `M` latent values from the conditional prior
   `p(z | source)`.
3. Decode each latent sample into target numerical values.
4. Inverse-transform decoded numerical values from gaussian-quantile scale back
   to the original data scale using target metadata.
5. For each row and target feature, build a prediction interval from the `M`
   generated values.
6. Count whether the observed target value falls inside the interval.
7. Use only rows where the observed target value is present.

## Prediction Interval

The default report includes 90%, 95%, and 99% prediction intervals.

For interval level `alpha`, use:

```text
lower_quantile = (1 - alpha) / 2
upper_quantile = 1 - lower_quantile
```

For `alpha = 0.90`, this is:

```text
lower = 5th percentile of generated samples
upper = 95th percentile of generated samples
```

Optional additional intervals:

- 50% interval: 25th to 75th percentile
- 80% interval: 10th to 90th percentile
- 95% interval: 2.5th to 97.5th percentile
- 99% interval: 0.5th to 99.5th percentile

## Metrics

Report metrics for each model and each numerical target feature:

- `n_observed`: number of rows with observed target values
- `interval_level`: requested interval level, such as `0.90`, `0.95`, or `0.99`
- `coverage`: fraction of observed values inside that prediction interval
- `coverage_error`: `coverage - interval_level`
- `mean_interval_width`: mean value of `upper - lower`
- `median_interval_width`: median value of `upper - lower`
- `mean_generated`: mean of per-row generated sample means
- `mean_observed`: mean of observed target values
- `mean_bias`: `mean_generated - mean_observed`

Coverage is:

```text
coverage = count(lower <= observed <= upper) / n_observed
```

## Interpretation

For an interval level `alpha`, ideal coverage is approximately `alpha`.

- Coverage below the requested level means the model distribution is too narrow, biased, or
  otherwise missing observed values too often.
- Coverage above the requested level means the interval may be too wide or the model may be
  over-uncertain.
- Coverage alone is not enough. Always inspect interval width too, because a
  very wide interval can obtain high coverage without being useful.
- Mean bias helps identify whether under-coverage is caused by systematic
  under-generation or over-generation.

## Comparison Rules

Compare models feature-by-feature within each disease/target group. Then compare
the aggregate behavior across all disease/target groups to detect models that
only work for a narrow subset of targets.

Preferred ordering:

1. Coverage closer to the requested interval level.
2. If coverage is similar, narrower interval width is better.
3. If both are similar, smaller absolute mean bias is better.

When a model has good coverage but poor AUROC, it may be calibrated for marginal
target values but weak at ranking disease risk. When a model has good AUROC but
poor coverage, it may rank samples well while producing miscalibrated target
distributions.

## Reproducibility

Each result file should record:

- checkpoint path
- dataset name
- target group
- source groups
- split
- numerical target feature
- number of prior samples `M`
- interval level
- random seed
- batch size
- mixed precision setting

Default evaluation settings:

```yaml
split: test
diseases: all
num_samples: 10000
interval_levels:
  - 0.90
  - 0.95
  - 0.99
seed: 42
```

# Evaluation Protocol

This document fixes the evaluation protocol used to compare the disease
prediction performance of the baseline booster and CVAE models.

## Metrics

For each disease, predictions are compared against observed test labels and
reported as:

- Sensitivity: `TP / (TP + FN)`
- Specificity: `TN / (TN + FP)`
- Balanced accuracy: `(sensitivity + specificity) / 2`

Thresholds are selected on the validation split only. Test sensitivity and
specificity are then computed once using the selected validation threshold.

## Boost Evaluation

For the CatBoost baseline:

1. Use the trained CatBoost model to produce disease-positive probabilities on
   the validation split.
2. Sweep candidate thresholds from the validation probabilities.
3. Select the threshold that maximizes validation balanced accuracy.
4. Apply this threshold to test probabilities.
5. Report test sensitivity and specificity.

The test split must not be used to choose the threshold.

## VAE Evaluation

For each CVAE disease model:

1. For each validation source row, generate `N` target samples from the prior.
2. Decode each generated target sample into target indicators.
   - Numerical target indicators use the decoded numerical value after inverse
     transformation back to the original scale.
   - For hepatitis B/C, each generated latent sample is decoded to categorical
     logits and converted with `softmax(cat_logits)`. The generated positive
     probability is the softmax probability of category `2` for `HE_hepaB` or
     `HE_hepaC`; the categorical output is not collapsed by argmax or sampled
     into a hard category before scoring.
3. For numerical-rule diseases, apply the disease rule to every generated
   target sample.
4. Treat the average generated positivity as the disease probability:
   - Numerical-rule diseases:
     `P(disease positive | source) = positive_generated_samples / N`.
   - Hepatitis B/C:
     `P(disease positive | source) = mean_N softmax(cat_logits)[category 2]`.
5. Select the threshold that maximizes validation balanced accuracy.
6. For each test source row, generate `N` target samples and compute the same
   disease probability.
7. Apply the validation-selected threshold to test probabilities.
8. Report test sensitivity and specificity.

`N` defaults to `100` but must remain configurable.

## Compared Models

The current comparison includes:

- CatBoost baseline
- Original CVAE

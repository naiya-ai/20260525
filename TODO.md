# TODO

## Goal

Implement a transformer-backbone CVAE for mixed tabular KNHANES/NHANES data.
The CatBoost baseline and the original CVAE remain the main comparison targets.
Evaluation should follow `EVALUATION.md`.

## Current Data State

- The dataset layout is expected to change.
- Harmonized and preprocessed data files are intentionally absent for now.
- Avoid baking assumptions about the old dataset artifact paths beyond the config layer.

## Tokenization

- `TabularFeatureTokenizer` is the shared first stage for EDDI-style encoders and transformer models.
- Its job is only to convert numerical and categorical tabular inputs into per-feature embeddings plus masks.
- It should not apply the EDDI shared NN or any sum/mean/max merge.
- Missing tokens are zeroed by `TabularFeatureTokenizer` before it returns
  `(tokens, masks)`.

## Missing Token Options

- `zero`: multiply tokens by `mask.unsqueeze(-1)` before the transformer and optionally exclude missing tokens with an attention mask.
- `learned`: use a learnable missing token per feature or per feature type; do not necessarily exclude missing tokens from attention.
- `keep`: preserve current raw token construction and rely on masks downstream.

Current default: `zero`. Add `learned` later as an ablation if needed.

## Transformer CVAE Architecture Questions

### Decoder Backbone

- Option A: transformer encoder + MLP decoder.
  - Recommended first implementation.
  - Keeps the decoder comparable to the original CVAE.
  - Reduces the number of moving parts while testing whether transformer source/posterior encoding helps.
- Option B: transformer encoder + transformer/query decoder.
  - Useful later for per-target-feature generation.
  - Requires target query tokens, per-feature output heads, and a clearer target-token decoding contract.

### Latent Projection

- Option A: add a learnable latent token.
  - Recommended first implementation.
  - Use the latent token output for `mu` and `logvar`.
  - Natural fit for both prior and posterior:
    - Prior input: source tokens + latent token.
    - Posterior input: source tokens + target context tokens + latent token.
- Option B: pool feature tokens by mean or sum, then project to `mu` and `logvar`.
  - Good ablation.
  - Closer to the original EDDI aggregation behavior.

## Proposed V1

- Done: source tokenizer uses `TabularFeatureTokenizer`.
- Done: target-context tokenizer uses the same class with separate parameters.
- Done: prior transformer:
  - input = source tokens + latent token
  - output latent token -> prior `mu`, prior `logvar`
  - latent token output can also serve as source condition
- Done: posterior transformer:
  - input = source tokens + target context tokens + latent token
  - output latent token -> posterior `mu`, posterior `logvar`
- Done: decoder:
  - keep MLP decoder for v1
  - input = sampled `z` + source condition
  - output = target numerical means + categorical logits

## Design Decisions Still Needed

- Done for v1: prior and posterior transformers use separate weights.
- Done for v1: source and target tokenizers use separate parameters.
- Whether to add feature-type embeddings or segment embeddings.
  - Useful if source and target tokens are concatenated.
- Whether attention masks should exclude missing tokens or allow learned missing tokens to participate.
- Whether to keep target-context encoding as flat one-hot/mask vectors for the original CVAE only, while transformer CVAE uses target tokens.
- Naming/config shape for selecting backbone:
  - Done: use `model.backbone: eddi | transformer`.

## Future Regularization Ablations

- Try `masked_target_dropout` later.
  - During posterior training, randomly hide observed target-context tokens while still reconstructing the original target.
  - Motivation: reduce posterior shortcut/copying from target tokens and make posterior training closer to the masked-target prior path.
  - Related ideas: denoising autoencoders, masked language modeling, tabular feature masking, modality/condition dropout.
  - Keep disabled for the current rerun.

## Sample Weight Cleanup

- Sample weighting is no longer part of the active experiment plan.
- Keep the implementation in model loss and dataset loading for possible future use.
- Done: removed default config usage and CLI overrides.
- Done: removed sample-weight-specific plotting/analysis scripts.

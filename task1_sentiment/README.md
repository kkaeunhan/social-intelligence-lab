# Task 1 Sentiment

This folder is for Task 1 sentiment-model experiments and training scripts.

## Files

- `model_candidates.py`: compare multiple model candidates on the same train/validation protocol.
- `train_sentiment.py`: train the selected final model and save it under `artifacts/task1_sentiment/`.

## Candidate Experiments

Run additional ELECTRA candidates with:

```bash
python task1_sentiment/model_candidates.py --models kr_electra koelectra_base
```

- `kr_electra`: `snunlp/KR-ELECTRA-discriminator`
- `koelectra_base`: `monologg/koelectra-base-v3-discriminator`

## Output Artifacts

Final Task 1 artifacts should be saved in:

```text
artifacts/task1_sentiment/
```

Suggested files:

- `final_model.joblib` for a scikit-learn pipeline.
- `model_config.json` for selected model settings.
- `metrics.json` for validation scores.

Keep the final inference path connected through `sentiment.py`.

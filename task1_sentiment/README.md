# Task 1 Sentiment

This folder is for Task 1 sentiment-model experiments and training scripts.

## Files

- `model_candidates.py`: compare multiple model candidates on the same train/validation protocol.
- `train_sentiment.py`: train the selected final model and save it under `artifacts/task1_sentiment/`.

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

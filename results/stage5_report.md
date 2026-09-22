# Stage 5 Tuning Report

- total_rows: 182814
- train_rows: 127962
- test_rows: 54852
- subtrain_rows: 91414
- val_rows: 36548
- classes: ['Final', 'Midterm 1', 'Midterm 2']

## Best Params
- n_estimators: 240
- max_depth: 7
- learning_rate: 0.07
- min_child_weight: 3
- subsample: 0.8
- colsample_bytree: 0.8

## Metrics
- validation: accuracy=0.595464, f1_macro=0.537905, auc_ovr_macro=0.830279
- test: accuracy=0.555732, f1_macro=0.440557, auc_ovr_macro=0.686663

## Outputs
- stage5_metrics.csv
- stage5_best_params.json
- stage5_confusion_xgboost_tuned.csv

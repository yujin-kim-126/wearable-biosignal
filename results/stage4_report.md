# Stage 4 Modeling Report

- total_rows: 182814
- train_rows: 127962
- test_rows: 54852
- train_sessions: 21
- test_sessions: 9
- classes: ['Final', 'Midterm 1', 'Midterm 2']

## Metrics
- baseline_logreg: accuracy=0.405637, f1_macro=0.411802, auc_ovr_macro=0.644350
- xgboost_fused: accuracy=0.444560, f1_macro=0.357185, auc_ovr_macro=0.638691

## Outputs
- stage4_metrics.csv
- stage4_confusion_baseline.csv
- stage4_confusion_xgboost.csv

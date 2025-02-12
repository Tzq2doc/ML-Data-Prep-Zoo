# ML Data Prep Zoo

A zoo of labelled datasets and ML models for data prep tasks.

## Task: ML Schema Inference

### Leaderboard

By submitting results, you acknowledge that your holdout test results([X_test.csv,y_test.csv]) are obtained purely by training on the training set([X_train.csv,y_train.csv]).

|        Model        | Overall Accuracy | Usable Numeric Accuracy | Usable-with-Extraction  Accuracy | Usable-Categorical Accuracy | Unusable Accuracy | Context-Specific Accuracy |
|:-------------------:|:----------------:|:-----------------------:|:--------------------------------:|:---------------------------:|:-----------------:|:-------------------------:|
| Random Forest       | 89.4             | 95.8                    | 79.7                             | 93.5                        | 74.3              | 83.4                      |
| k-NN                | 88.8             | 95.6                    | 77.6                             | 87.9                        | 79                | 85.6                      |
| Character-level CNN | 88.3             | 93.6                    | 83.9                             | 88.2                        | 83.2              | 83.2                      |
| RBF-SVM             | 87.4             | 94                      | 84.6                             | 89.1                        | 76                | 80.7                      |
| Logistic Regression | 86               | 94.5                    | 73.4                             | 86.8                        | 78.4              | 77.2                      |

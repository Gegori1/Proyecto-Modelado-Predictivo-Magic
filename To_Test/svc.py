#%% import libraries
import pickle
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)

#%% =============================================================================
### Load pickles
# =============================================================================

svc_path = Path(__file__).parent.parent / "pickles" / "SVC"

## Load data
with open(svc_path / "data_svc.pkl", "rb") as f:
    data = pickle.load(f)

with open(svc_path / "data_svc_cv.pkl", "rb") as f:
    data_cv = pickle.load(f)

## Load model
with open(svc_path / "model_svc.pkl", "rb") as f:
    model = pickle.load(f)

with open(svc_path / "model_svc_cv.pkl", "rb") as f:
    model_cv = pickle.load(f)

#%% =============================================================================
### Assign variables
# =============================================================================

X_train, X_test, y_train, y_test, y_predict, y_predict_test = data

X_train_cv, X_test_cv, y_train_cv, y_test_cv, y_predict_cv, y_predict_test_cv = data_cv

#%% =============================================================================
### Predict
# =============================================================================

y_predict = model.predict(X_train)
y_predict_test = model.predict(X_test)

y_predict_cv = model_cv.predict(X_train_cv)
y_predict_test_cv = model_cv.predict(X_test_cv)

#%% =============================================================================
### Metrics
# =============================================================================

## Metrics without cv
print("Model without transformation:")
print(f"Accuracy train: {accuracy_score(y_train, y_predict):.3f}")
print(f"Accuracy test: {accuracy_score(y_test, y_predict_test):.3f}")
print(f"Recall train: {recall_score(y_train, y_predict):.3f}")
print(f"Recall test: {recall_score(y_test, y_predict_test):.3f}")
print(f"Precision train: {precision_score(y_train, y_predict):.3f}")
print(f"Precision test: {precision_score(y_test, y_predict_test):.3f}")
print(f"F1 train: {f1_score(y_train, y_predict):.3f}")
print(f"F1 test: {f1_score(y_test, y_predict_test):.3f}")
print(f"Confusion matrix train: \n{confusion_matrix(y_train, y_predict)}")
print(f"Confusion matrix test: \n{confusion_matrix(y_test, y_predict_test)}")

print("\n")
# Metrics with cv
print("Model with transformation:")
print(f"Accuracy train cv: {accuracy_score(y_train_cv, y_predict_cv):.3f}") 
print(f"Accuracy test cv: {accuracy_score(y_test_cv, y_predict_test_cv):.3f}")
print(f"Recall train cv: {recall_score(y_train_cv, y_predict_cv):.3f}")
print(f"Recall test cv: {recall_score(y_test_cv, y_predict_test_cv):.3f}")
print(f"Precision train cv: {precision_score(y_train_cv, y_predict_cv):.3f}")
print(f"Precision test cv: {precision_score(y_test_cv, y_predict_test_cv):.3f}")
print(f"F1 train cv: {f1_score(y_train_cv, y_predict_cv):.3f}")
print(f"F1 test cv: {f1_score(y_test_cv, y_predict_test_cv):.3f}")
print(f"Confusion matrix train cv: \n{confusion_matrix(y_train_cv, y_predict_cv)}")
print(f"Confusion matrix test cv: \n{confusion_matrix(y_test_cv, y_predict_test_cv)}")


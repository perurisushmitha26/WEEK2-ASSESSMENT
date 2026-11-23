# =====================================================
# TASK 4: EVALUATION AND TUNING
# =====================================================


import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy.stats import randint, uniform


from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyRegressor


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


from xgboost import XGBClassifier


warnings.filterwarnings("ignore")




# =====================================================
# 1. FOLDER PATH
# =====================================================


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


DATA_PATH = os.path.join(BASE_DIR, "data", "loan_data.csv")


OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)




# =====================================================
# 2. LOAD DATASET
# =====================================================


df = pd.read_csv(DATA_PATH)


print("\n========== DATASET ==========")
print(df.head())
print("Shape:", df.shape)
print(df.info())


print("\nMissing Values:")
print(df.isnull().sum())




# =====================================================
# 3. FEATURE ENGINEERING
# =====================================================


required_columns = [
    "application_date",
    "income",
    "loan_amount",
    "credit_score",
    "dependents",
    "employment_type",
    "marital_status",
    "education",
    "loan_approved"
]


missing_columns = [
    col for col in required_columns if col not in df.columns
]


if missing_columns:
    raise ValueError(f"Missing columns in CSV: {missing_columns}")


df["application_date"] = pd.to_datetime(
    df["application_date"], errors="coerce"
)


df["year"] = df["application_date"].dt.year
df["month"] = df["application_date"].dt.month
df["day"] = df["application_date"].dt.day


df["income_loan_ratio"] = (
    df["income"] / df["loan_amount"].replace(0, np.nan)
)


df["log_loan_amount"] = np.log1p(
    df["loan_amount"].clip(lower=0)
)


df.drop(columns=["application_date"], inplace=True)




# =====================================================
# 4. CLASSIFICATION TARGET
# =====================================================


TARGET = "loan_approved"


df = df.dropna(subset=[TARGET]).copy()


X = df.drop(columns=[TARGET]).copy()


y = df[TARGET].copy()


if y.dtype == "object":
    y = y.astype(str).str.strip().str.lower()


    y = y.map({
        "yes": 1,
        "no": 0,
        "approved": 1,
        "rejected": 0,
        "true": 1,
        "false": 0
    })


    if y.isnull().any():
        raise ValueError("Invalid loan_approved labels.")


y = y.astype(int)


if set(y.unique()) != {0, 1}:
    raise ValueError("Target must contain both 0 and 1.")


print("\n========== CLASS DISTRIBUTION ==========")
print(y.value_counts())




# =====================================================
# 5. TRAIN VALIDATION TEST SPLIT
# =====================================================


X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val
)


print("\nTraining:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)




# =====================================================
# 6. PREPROCESSING
# =====================================================


numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()


categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()


numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])




# =====================================================
# 7. CLASSIFICATION MODELS
# =====================================================


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),


    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    ),


    "Random Forest": RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )
}


fitted_models = {}


print("\n========== CLASSIFICATION MODELS ==========")


for name, model in models.items():


    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])


    pipeline.fit(X_train, y_train)


    fitted_models[name] = pipeline


    predictions = pipeline.predict(X_val)


    print("\nModel:", name)


    print("Accuracy:",
          accuracy_score(y_val, predictions))


    print("Precision:",
          precision_score(y_val, predictions, zero_division=0))


    print("Recall:",
          recall_score(y_val, predictions, zero_division=0))


    print("F1 Score:",
          f1_score(y_val, predictions, zero_division=0))


    print("Confusion Matrix:")
    print(confusion_matrix(y_val, predictions))




# =====================================================
# 8. RANDOM FOREST GRID SEARCH CV
# =====================================================


print("\n========== RANDOM FOREST GRID SEARCH ==========")


rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),


    ("classifier", RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ))
])


rf_parameters = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [5, 10, 15],
    "classifier__min_samples_leaf": [1, 2, 4]
}


grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_parameters,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)


grid_search.fit(X_train, y_train)


best_rf = grid_search.best_estimator_


print("\nBest Random Forest Parameters:")
print(grid_search.best_params_)


print("Best Random Forest Score:")
print(grid_search.best_score_)


fitted_models["Random Forest"] = best_rf




# =====================================================
# 9. XGBOOST RANDOMIZED SEARCH CV
# =====================================================


print("\n========== XGBOOST RANDOMIZED SEARCH ==========")


xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),


    ("classifier", XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_jobs=1
    ))
])


xgb_parameters = {
    "classifier__n_estimators": randint(100, 501),
    "classifier__max_depth": randint(3, 8),
    "classifier__learning_rate": uniform(0.01, 0.19),
    "classifier__subsample": uniform(0.7, 0.3),
    "classifier__colsample_bytree": uniform(0.7, 0.3)
}


random_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=xgb_parameters,
    n_iter=15,
    cv=3,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
    verbose=1
)


random_search.fit(X_train, y_train)


best_xgb = random_search.best_estimator_


print("\nBest XGBoost Parameters:")
print(random_search.best_params_)


print("Best XGBoost Score:")
print(random_search.best_score_)


fitted_models["XGBoost"] = best_xgb




# =====================================================
# 10. EVALUATE EVERY CLASSIFIER
# =====================================================


print("\n========== ALL CLASSIFIER RESULTS ==========")


validation_results = {}


for name, model in fitted_models.items():


    predictions = model.predict(X_val)


    probabilities = model.predict_proba(X_val)[:, 1]


    validation_results[name] = {
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_val, predictions, zero_division=0
        ),
        "AUC": roc_auc_score(y_val, probabilities)
    }


    print("\nModel:", name)


    for metric, value in validation_results[name].items():
        print(f"{metric}: {value:.4f}")


    print("Confusion Matrix:")
    print(confusion_matrix(y_val, predictions))




# =====================================================
# 11. ROC CURVE OF THREE MODELS
# =====================================================


plt.figure(figsize=(9, 6))


roc_models = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost"
]


for name in roc_models:


    model = fitted_models[name]


    probabilities = model.predict_proba(X_val)[:, 1]


    fpr, tpr, thresholds = roc_curve(
        y_val, probabilities
    )


    auc_value = roc_auc_score(
        y_val, probabilities
    )


    plt.plot(
        fpr,
        tpr,
        label=f"{name} AUC={auc_value:.3f}"
    )


plt.plot([0, 1], [0, 1], linestyle="--")


plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")


plt.title("ROC Curve Comparison")


plt.legend()
plt.grid(True)


plt.savefig(
    os.path.join(OUTPUT_DIR, "roc_curves.png")
)


plt.show()




# =====================================================
# 12. SELECT BEST MODEL
# =====================================================


best_model_name = max(
    validation_results,
    key=lambda name: validation_results[name]["AUC"]
)


best_model = fitted_models[best_model_name]


print("\n========== BEST MODEL ==========")


print("Best Model:", best_model_name)


print("ROC-AUC:",
      validation_results[best_model_name]["AUC"])




# =====================================================
# 13. DECISION THRESHOLD
# =====================================================


probabilities = best_model.predict_proba(X_val)[:, 1]


precision_values, recall_values, thresholds = (
    precision_recall_curve(y_val, probabilities)
)


valid_indices = np.where(
    recall_values[:-1] >= 0.75
)[0]


if len(valid_indices) > 0:


    best_index = valid_indices[
        np.argmax(precision_values[valid_indices])
    ]


    selected_threshold = thresholds[best_index]


    print("\n========== DECISION THRESHOLD ==========")


    print("Threshold:", selected_threshold)


    print("Recall:", recall_values[best_index])


    print("Precision:", precision_values[best_index])


else:
    raise ValueError(
        "No threshold achieved at least 75% recall."
    )




# =====================================================
# 14. REGRESSION EVALUATION
# =====================================================


print("\n========== REGRESSION EVALUATION ==========")


regression_features = df.drop(
    columns=[
        "loan_amount",
        "loan_approved",
        "log_loan_amount",
        "income_loan_ratio"
    ]
)


regression_target = df["loan_amount"]


reg_X_train_val, reg_X_test, reg_y_train_val, reg_y_test = (
    train_test_split(
        regression_features,
        regression_target,
        test_size=0.20,
        random_state=42
    )
)


reg_X_train, reg_X_val, reg_y_train, reg_y_val = (
    train_test_split(
        reg_X_train_val,
        reg_y_train_val,
        test_size=0.25,
        random_state=42
    )
)


reg_numeric = regression_features.select_dtypes(
    include=np.number
).columns.tolist()


reg_categorical = regression_features.select_dtypes(
    exclude=np.number
).columns.tolist()


reg_preprocessor = ColumnTransformer([
    ("num", numeric_transformer, reg_numeric),
    ("cat", categorical_transformer, reg_categorical)
])


reg_models = {
    "Linear Regression": LinearRegression(),
    "Mean Baseline": DummyRegressor(strategy="mean")
}


reg_results = {}


for name, model in reg_models.items():


    pipeline = Pipeline([
        ("preprocessor", reg_preprocessor),
        ("regressor", model)
    ])


    pipeline.fit(reg_X_train, reg_y_train)


    predictions = pipeline.predict(reg_X_val)


    mae = mean_absolute_error(
        reg_y_val, predictions
    )


    rmse = np.sqrt(
        mean_squared_error(reg_y_val, predictions)
    )


    r2 = r2_score(
        reg_y_val, predictions
    )


    reg_results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


    print("\nModel:", name)


    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R-Squared:", round(r2, 4))




# =====================================================
# 15. FINAL TEST EVALUATION
# =====================================================


print("\n========== FINAL TEST EVALUATION ==========")


test_probabilities = best_model.predict_proba(X_test)[:, 1]


test_predictions = (
    test_probabilities >= selected_threshold
).astype(int)


print("Best Model:", best_model_name)


print("Accuracy:",
      accuracy_score(y_test, test_predictions))


print("Precision:",
      precision_score(y_test, test_predictions, zero_division=0))


print("Recall:",
      recall_score(y_test, test_predictions, zero_division=0))


print("F1 Score:",
      f1_score(y_test, test_predictions, zero_division=0))


print("ROC-AUC:",
      roc_auc_score(y_test, test_probabilities))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))


print("\nClassification Report:")
print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))




# =====================================================
# 16. SAVE BEST MODEL
# =====================================================


model_path = os.path.join(
    MODEL_DIR,
    "best_loan_classifier.joblib"
)


joblib.dump(best_model, model_path)


print("\nBest Model Saved Successfully!")




# =====================================================
# 17. SAVE RESULTS
# =====================================================


pd.DataFrame(validation_results).T.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "classification_metrics.csv"
    )
)


pd.DataFrame(reg_results).T.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "regression_metrics.csv"
    )
)


print("\nAll Results Saved Successfully!")


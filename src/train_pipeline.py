import logging
import os
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

random_state = 42  # Ensure that pipeline is reproducible.

log_format = (
    "[%(asctime)s] - p%(process)s %(name)s %(lineno)d - %(levelname)s:%(message)s"
)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format=log_format,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


def create_training_pipeline(data_dir, feature_columns, target_column):

    """Test"""

    # Load Dataset
    df_raw = pd.read_csv(
        os.path.join(data_dir, "raw_stroke_records.csv")
    )  # Load raw dataset as Pandas DataFrame.
    logger.info(f"Raw Dataset Number of Records: {len(df_raw)}")

    # Process Dataset
    df_processed = df_raw.drop("id", axis=1).reset_index(drop=True)  # Drop id column.

    df_processed = df_processed[
        df_processed["gender"].isin(["Male", "Female"])
    ].reset_index(
        drop=True
    )  # Ensure gender only corresponds to Male and Female.
    logger.info(f"Processed Dataset Number of Rows: {len(df_processed)}")

    df_processed.to_csv(
        "../data/processed/stroke_records.csv", index=False
    )  # Export processed Pandas DataFrame as .csv file.

    X = df_processed[feature_columns]  # Select training features.
    y = df_processed[target_column]  # Select predictor variable.

    # Create Pipeline
    numeric_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns  # Evaluate numeric columns using dtype.
    logger.info(f"Numeric Columns: {numeric_columns}")

    categorical_columns = X.select_dtypes(
        include=["object", "bool"]
    ).columns  # Evaluate categorical columns using dtype.
    logger.info(f"Categorical Columns: {numeric_columns}")

    preprocess_pipeline = ColumnTransformer(
        [
            (
                "num_imputer",
                SimpleImputer(missing_values=np.nan, strategy="median"),
                numeric_columns,
            ),
            ("categorical_encoder", OneHotEncoder(), categorical_columns),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess_pipeline),
            ("classifier", RandomForestClassifier()),
        ]
    )

    # Define Hyperparameter Space
    param_grid = {
        "classifier__max_depth": [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, None],
        "classifier__criterion": ["gini", "entropy"],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_grid,
        n_iter=20,
        cv=KFold(n_splits=5, shuffle=True, random_state=random_state),
    )

    # Train Pipeline Using Cross-Validation
    logger.info(f"Starting Pipeline Training.")
    train_acc, val_acc = [], []  # Define empty lists.
    for train_ind, val_ind in KFold(
        n_splits=5, shuffle=True, random_state=random_state
    ).split(X, y):
        X_train, y_train = X.iloc[train_ind], y[train_ind]  # Select fold training data.
        X_val, y_val = X.iloc[val_ind], y[val_ind]  # Select fold validation data.

        search.fit(X_train, y_train)  # Fit model using training data.

        y_hat_train = search.predict(X_train)
        train_acc.append(
            accuracy_score(y_train, y_hat_train)
        )  # Evaluate fold train accuracy.

        y_hat_val = search.predict(X_val)
        val_acc.append(
            accuracy_score(y_val, y_hat_val)
        )  # Evaluate fold validation accuracy.

    # Assess Model Performance
    mean_train_acc = np.round(
        np.mean(train_acc), 4
    )  # Evaluate average training accuracy.
    logger.info(f"Training Accuracy: {mean_train_acc}")

    mean_val_acc = np.round(
        np.mean(val_acc), 4
    )  # Evaluate average validation accuracy.
    logger.info(f"Validation Accuracy: {mean_val_acc}")

    logger.info(f"Optimized Hyperparameters: {search.best_params_}")

    # Export Model
    if not os.path.exists("pipelines"):
        os.makedirs("pipelines")
    joblib.dump(
        search.best_estimator_, f"pipelines/RF_A_{mean_val_acc}.joblib"
    )  # NOTE: GridSearchCV returns model fitted to full dataset (see: https://stackoverflow.com/questions/34143829/sklearn-how-to-save-a-model-created-from-a-pipeline-and-gridsearchcv-using-jobli).

    logger.info(f"Exported Pipeline: pipelines/RF_A_{mean_val_acc}.joblib")


if __name__ == "__main__":
    create_training_pipeline(
        data_dir="../data/raw/",
        feature_columns=[
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "avg_glucose_level",
            "bmi",
            "smoking_status",
        ],
        target_column="stroke",
    )

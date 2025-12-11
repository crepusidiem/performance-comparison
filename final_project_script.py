import warnings

warnings.filterwarnings("ignore")

import os
import ssl
import urllib.request
from io import BytesIO

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_svmlight_file


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

RESULTS_PATH = "final_project_perfect_results.csv"


def load_existing_results(results_path):
    """
    If there is already a CSV，read and return as DataFrame；
    otherwise return an empty DataFrame with the correct columns.
    """
    if os.path.exists(results_path):
        print(f"[Resume] Found existing results file: {results_path}")
        df = pd.read_csv(results_path)
        # Make sure all required columns are present
        required_cols = [
            "Dataset",
            "Classifier",
            "Train %",
            "Run",
            "Train Acc",
            "CV Acc",
            "Test Acc",
            "Best Params",
        ]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Existing results file missing column: {col}")
        return df
    else:
        print(f"[Resume] No existing results file. Starting fresh.")
        return pd.DataFrame(
            columns=[
                "Dataset",
                "Classifier",
                "Train %",
                "Run",
                "Train Acc",
                "CV Acc",
                "Test Acc",
                "Best Params",
            ]
        )


def is_already_done(existing_df, dataset_name, train_pct, run, clf_name):
    """
    Check to see if a combination (Dataset, Train %, Run, Classifier) already exists
    """
    if existing_df.empty:
        return False

    mask = (
        (existing_df["Dataset"] == dataset_name)
        & (existing_df["Classifier"] == clf_name)
        & (existing_df["Train %"] == int(train_pct))
        & (existing_df["Run"] == int(run))
    )
    return mask.any()


def load_datasets():
    """
    Load the 4 datasets of Adult / Spambase / Letter / Cod-RNA
    return: datasets = {name: (X, y)}
    """
    datasets = {}

    print("Loading Adult dataset...")
    adult = fetch_ucirepo(id=2)
    X_adult = adult.data.features
    y_adult = (adult.data.targets.values.ravel() == ">50K").astype(int)
    datasets["Adult"] = (X_adult, y_adult)

    print("Loading Spambase dataset...")
    spambase = fetch_ucirepo(id=94)
    X_spam = spambase.data.features
    y_spam = spambase.data.targets.values.ravel()
    datasets["Spambase"] = (X_spam, y_spam)

    print("Loading Letter dataset (A vs rest)...")
    letter = fetch_ucirepo(id=59)
    X_letter = letter.data.features
    y_letter = (letter.data.targets.values.ravel() == "A").astype(int)
    datasets["Letter"] = (X_letter, y_letter)

    print("Downloading Cod-RNA (libsvm format)...", end=" ", flush=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna",
        context=ctx,
    ) as resp:
        data = resp.read()

    X_cod_full, y_cod_full = load_svmlight_file(BytesIO(data))
    X_cod_full = X_cod_full.toarray()

    # subsample 100000 rows for speed considerations
    np.random.seed(RANDOM_STATE)
    idx = np.random.permutation(X_cod_full.shape[0])[:100000]
    X_cod = X_cod_full[idx]
    y_cod = y_cod_full[idx].astype(int)
    y_cod = np.where(y_cod == -1, 0, 1)  # explicitly map -1→0, +1→1 for XGBoost
    print(f"Done → {X_cod.shape[0]:,} samples × {X_cod.shape[1]} features")

    del X_cod_full, y_cod_full
    datasets["Cod-RNA"] = (X_cod, y_cod)

    return datasets


def build_models_and_grids():
    """
    Define classifier and their respective hyperparameter spaces，Use RandomizedSearch for heavy models
    """
    classifiers = {
        "SVM-RBF": SVC(probability=True, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        "NeuralNet": MLPClassifier(
            max_iter=600, early_stopping=True, random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(),
    }

    # Define hyperparameter grids
    param_grids = {
        "SVM-RBF": {"clf__C": [1, 10], "clf__gamma": ["scale", 0.1]},
        "RandomForest": {"clf__n_estimators": [200, 500], "clf__max_depth": [None, 30]},
        "XGBoost": {
            "clf__n_estimators": [200, 400],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [6, 10],
        },
        "NeuralNet": {
            "clf__hidden_layer_sizes": [(100,), (200, 100)],
            "clf__alpha": [0.0001, 0.001],
        },
        "KNN": {"clf__n_neighbors": [5, 11], "clf__weights": ["uniform", "distance"]},
    }

    # Use RandomizedSearchCV for the following heavy models
    random_search_models = {"SVM-RBF", "XGBoost", "NeuralNet"}

    return classifiers, param_grids, random_search_models


def run_experiments(
    datasets,
    classifiers,
    param_grids,
    random_search_models,
    partitions=(0.2, 0.5, 0.8),
    n_runs=3,
    n_random_iters=10,
    cv_folds=3,
    results_path=RESULTS_PATH,
):

    # Read existing results if any, otherwise initialize it
    existing_df = load_existing_results(results_path)
    if not existing_df.empty:
        results = existing_df.to_dict(orient="records")
    else:
        results = []

    for dataset_name, (X_raw, y) in datasets.items():
        print(f"\n=== Processing {dataset_name} ===")

        # categorize features
        if isinstance(X_raw, pd.DataFrame):
            cat_features = X_raw.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            num_features = X_raw.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
        else:
            # Default to all numeric if not DataFrame
            cat_features = []
            num_features = list(range(X_raw.shape[1]))

        # Define preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_features),
                (
                    "cat",
                    OneHotEncoder(
                        drop="first", handle_unknown="ignore", sparse_output=False
                    ),
                    cat_features,
                ),
            ],
            remainder="passthrough",
        )

        for train_ratio in partitions:
            train_pct = int(train_ratio * 100)
            for run in range(1, n_runs + 1):
                print(f"  Train {train_pct}% | Run {run} ...", flush=True)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_raw,
                    y,
                    train_size=train_ratio,
                    stratify=y,
                    random_state=RANDOM_STATE + run,
                )

                for clf_name, clf in classifiers.items():
                    # Check if this combination is already done
                    if is_already_done(
                        existing_df, dataset_name, train_pct, run, clf_name
                    ):
                        print(f"    -> {clf_name}: already done, skipping.")
                        continue

                    print(f"    -> {clf_name}: tuning...", end=" ", flush=True)

                    pipe = Pipeline([("preprocessor", preprocessor), ("clf", clf)])

                    param_grid = param_grids[clf_name]

                    # RandomizedSearchCV for heavy models
                    if clf_name in random_search_models:
                        search = RandomizedSearchCV(
                            estimator=pipe,
                            param_distributions=param_grid,
                            n_iter=n_random_iters,
                            cv=cv_folds,
                            scoring="accuracy",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        )
                    else:
                        # GridSearchCV for others
                        search = GridSearchCV(
                            estimator=pipe,
                            param_grid=param_grid,
                            cv=cv_folds,
                            scoring="accuracy",
                            n_jobs=-1,
                        )

                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_

                    train_acc = accuracy_score(y_train, best_model.predict(X_train))
                    test_acc = accuracy_score(y_test, best_model.predict(X_test))
                    cv_acc = search.best_score_

                    row = {
                        "Dataset": dataset_name,
                        "Classifier": clf_name,
                        "Train %": train_pct,
                        "Run": run,
                        "Train Acc": round(train_acc, 4),
                        "CV Acc": round(cv_acc, 4),
                        "Test Acc": round(test_acc, 4),
                        "Best Params": search.best_params_,
                    }

                    results.append(row)

                    # Update existing_df，so we can check for duplicates next time
                    existing_df = pd.DataFrame(results)

                    # Write checkpoint to CSV for resuming later
                    existing_df.to_csv(results_path, index=False)

                    print("done. [checkpoint saved]")

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)
    print(f"\nAll experiments finished. Final results saved to {results_path}")

    return results_df


def summarize_results(results_df):
    """
    Print summary of results
    """
    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    # Averate Test Acc over runs
    summary = (
        results_df.groupby(["Dataset", "Classifier", "Train %"])["Test Acc"]
        .mean()
        .round(4)
        .unstack()
    )

    # Reorder columns
    cols_order = ["SVM-RBF", "KNN", "NeuralNet", "XGBoost", "RandomForest"]
    summary = summary[[c for c in cols_order if c in summary.columns]]

    print("\nTest Accuracy (averaged over runs):")
    print(summary)

    # Best performer per Dataset & Training Size
    print("\nOverall Best Performer per Dataset & Training Size:")
    best_per_group_idx = results_df.groupby(["Dataset", "Train %"])["Test Acc"].idxmax()
    best_per_group = results_df.loc[best_per_group_idx]
    best_per_group = best_per_group[
        ["Dataset", "Train %", "Classifier", "Test Acc"]
    ].sort_values(["Dataset", "Train %", "Classifier"])

    print(best_per_group.to_string(index=False))


# MAIN EXECUTION
def main():
    print("Script started...")

    datasets = load_datasets()
    classifiers, param_grids, random_search_models = build_models_and_grids()

    results_df = run_experiments(
        datasets=datasets,
        classifiers=classifiers,
        param_grids=param_grids,
        random_search_models=random_search_models,
        partitions=(0.2, 0.5, 0.8),
        n_runs=3,
        n_random_iters=10,
        cv_folds=3,
        results_path=RESULTS_PATH,
    )

    summarize_results(results_df)


if __name__ == "__main__":
    main()

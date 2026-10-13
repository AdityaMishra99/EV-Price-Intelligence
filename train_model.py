#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

TARGET = "price_usd"

DROP = [
    TARGET,
    "market_segment",
    "segment_from_price",
    "torque_nm",
    "warranty_years"
]

data_path = r"C:/Users/adimi/Downloads/Data Analytics/EV Market/Data/ev_market_2026.csv"

market = pd.read_csv(data_path)

# Target: EV price
y = market[TARGET]

# Features: columns used to predict EV price
X = market.drop(
    columns=[col for col in DROP if col in market.columns]
)

print("Dataset loaded successfully!")
print("Feature shape:", X.shape)
print("Target shape:", y.shape)
print("Features:", X.columns.tolist())


# In[ ]:


import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "price_usd"
# market_segment and segment_from_price are derived from price (target leakage).
# torque_nm duplicates horsepower; warranty_years added no signal in testing.
DROP = [TARGET, "market_segment", "segment_from_price", "torque_nm", "warranty_years"]


def build_hierarchy(df: pd.DataFrame, columns=("brand", "model", "variant")):
    """Nested {brand: {model: [variants]}} so the app only offers models that belong to a brand."""
    cols = [c for c in columns if c in df.columns]
    if len(cols) < 2:
        return None
    tree = {}
    for row in df[cols].astype(str).drop_duplicates().itertuples(index=False):
        node = tree
        for i, value in enumerate(row[:-1]):
            node = node.setdefault(value, [] if i == len(cols) - 2 else {})
        node.append(row[-1])

    def tidy(node):
        if isinstance(node, dict):
            return {k: tidy(node[k]) for k in sorted(node)}
        return sorted(set(node))

    return {"columns": cols, "tree": tidy(tree)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:/Users/adimi/Downloads/Data Analytics/EV Market/Data/ev_market_2026.csv")
    ap.add_argument("--out", default="ev_price_model.joblib")
    ap.add_argument("--n-iter", type=int, default=20, help="random-search candidates")
    args, _ = ap.parse_known_args()

    market = pd.read_csv(args.data)
    y = market[TARGET]
    X = market.drop(columns=[c for c in DROP if c in market.columns])

    # Safety net: no remaining feature may look derived from the price.
    suspect = [c for c in X.columns if "price" in c.lower()]
    if suspect:
        raise SystemExit(f"Possible target leakage, remove these features first: {suspect}")

    cat_cols = list(X.select_dtypes(include=["object", "category", "string"]).columns)
    num_cols = list(X.select_dtypes(include=["number"]).columns)
    print(f"{len(X.columns)} features | categorical: {cat_cols} | numeric: {num_cols}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def make(**params) -> Pipeline:
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ])
        return Pipeline([("preprocessor", pre),
                         ("model", RandomForestRegressor(random_state=42, n_jobs=-1, **params))])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline
    baseline = make(n_estimators=200)
    base_cv_mae = -cross_val_score(baseline, X_train, y_train, cv=cv,
                                   scoring="neg_mean_absolute_error").mean()
    baseline.fit(X_train, y_train)

    # Tuned
    search = RandomizedSearchCV(
        make(),
        param_distributions={
            "model__n_estimators": [100, 150, 200, 300],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [1.0, "sqrt", 0.5],
        },
        n_iter=args.n_iter, scoring="neg_mean_absolute_error", cv=cv,
        random_state=42, n_jobs=-1,
    ).fit(X_train, y_train)
    tuned, tuned_cv_mae = search.best_estimator_, -search.best_score_

    # Choose by cross-validation (training data only), then test once.
    use_tuned = tuned_cv_mae < base_cv_mae
    final, name = (tuned, "Tuned Random Forest") if use_tuned else (baseline, "Baseline Random Forest")

    def score(model):
        p = model.predict(X_test)
        return p, {"mae": float(mean_absolute_error(y_test, p)),
                   "rmse": float(mean_squared_error(y_test, p) ** 0.5),
                   "r2": float(r2_score(y_test, p))}

    _, base_s = score(baseline)
    _, tuned_s = score(tuned)
    pred, final_s = score(final)

    print(f"\nCV MAE (train only): baseline {base_cv_mae:,.0f} | tuned {tuned_cv_mae:,.0f}")
    print(f"Test  baseline: {base_s}\nTest  tuned:    {tuned_s}")
    print(f"\nSelected: {name}  ->  MAE {final_s['mae']:,.0f} | RMSE {final_s['rmse']:,.0f} | R2 {final_s['r2']:.4f}")

    bundle = {
        "model": final,
        "feature_columns": list(X.columns),
        "categorical_options": {c: sorted(market[c].dropna().astype(str).unique().tolist()) for c in cat_cols},
        "numeric_defaults": {c: float(market[c].median()) for c in num_cols},
        "numeric_ranges": {c: {"min": float(market[c].min()), "max": float(market[c].max()),
                               "median": float(market[c].median())} for c in num_cols},
        "hierarchy": build_hierarchy(market),
        "metrics": {**final_s, "n_train": int(len(X_train))},
        "evaluation": {"y_true": y_test.tolist(), "y_pred": pred.tolist()},
    }
    joblib.dump(bundle, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()


# In[ ]:





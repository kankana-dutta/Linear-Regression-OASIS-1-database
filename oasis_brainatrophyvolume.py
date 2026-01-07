#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OASIS (tabular) Linear Regression Project
----------------------------------------
Task (your choice):
  Predict nWBV (normalized whole brain volume; proxy for atrophy)
  from: Age + eTIV + ASF + Hand

Why this is a good regression demo:
  - nWBV is available for most subjects (usually far less missing than MMSE)
  - nWBV is biologically linked to age + head-size/scale measures
  - Linear regression often gives cleaner plots and higher R^2 than MMSE prediction

Evaluation:
  - MSE
  - R^2

Outputs:
  - Train/Test MSE and R^2
  - Coefficient table (standardized)
  - Predicted vs True scatter plot
  - Residual plot

Requirements:
  pip install pandas numpy scikit-learn matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score


# -------------------------
# USER SETTINGS
# -------------------------
META_CSV  = r"C:\Users\witty\Desktop\Neural networks\oasis_1_data.csv"

TARGET    = "nWBV"
NUM_FEATS = ["Age", "eTIV", "ASF"]
CAT_FEATS = ["Hand"]   # categorical (e.g., R/L/A). We'll one-hot encode it.

TEST_SIZE = 0.20
SEED      = 42

# -------------------------
# 1) Load data
# -------------------------
if not os.path.exists(META_CSV):
    raise FileNotFoundError(f"Cannot find META_CSV: {META_CSV}")

df = pd.read_csv(META_CSV)

required = [TARGET] + NUM_FEATS + CAT_FEATS
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    raise ValueError(
        "Your CSV is missing required columns:\n"
        f"{missing_cols}\n\n"
        f"Available columns are:\n{list(df.columns)}"
    )

# Keep only required columns
data = df[required].copy()

# Numeric conversion
for c in NUM_FEATS + [TARGET]:
    data[c] = pd.to_numeric(data[c], errors="coerce")

# Clean/standardize Hand strings (keep as category)
data["Hand"] = data["Hand"].astype(str).str.strip()
# Common cleanup: if blank becomes 'nan' string
data.loc[data["Hand"].str.lower().isin(["nan", "none", ""]), "Hand"] = np.nan

# Drop rows where target is missing (cannot train without y)
data = data.dropna(subset=[TARGET]).reset_index(drop=True)

if len(data) < 30:
    raise RuntimeError(
        f"Too few labeled rows after filtering (N={len(data)}). "
        "Check if nWBV is missing for many rows."
    )

X = data[NUM_FEATS + CAT_FEATS]
y = data[TARGET].astype(float).values


# -------------------------
# 2) Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=SEED
)


# -------------------------
# 3) Preprocessing:
#    - Numeric: median impute + standardize
#    - Categorical Hand: most_frequent impute + one-hot encode
# -------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, NUM_FEATS),
        ("cat", categorical_transformer, CAT_FEATS),
    ],
    remainder="drop"
)

# -------------------------
# 4) Linear regression model
# -------------------------
linreg = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LinearRegression())
])

linreg.fit(X_train, y_train)

y_pred_train = linreg.predict(X_train)
y_pred_test  = linreg.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
r2_train  = r2_score(y_train, y_pred_train)

mse_test  = mean_squared_error(y_test, y_pred_test)
r2_test   = r2_score(y_test, y_pred_test)

print("\n==============================")
print("Linear Regression (Baseline)")
print("==============================")
print(f"Target:   {TARGET}")
print(f"Numeric features: {NUM_FEATS}")
print(f"Categorical features: {CAT_FEATS}")
print("------------------------------")
print(f"Train: MSE={mse_train:.6f} | R^2={r2_train:.3f}")
print(f"Test : MSE={mse_test:.6f} | R^2={r2_test:.3f}")

# -------------------------
# 5) Ridge regression (robustness)
# -------------------------
ridge = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", Ridge(alpha=1.0, random_state=SEED))
])

ridge.fit(X_train, y_train)
y_ridge_test = ridge.predict(X_test)

mse_ridge = mean_squared_error(y_test, y_ridge_test)
r2_ridge  = r2_score(y_test, y_ridge_test)

print("\n==============================")
print("Ridge Regression (Robustness)")
print("==============================")
print(f"Test : MSE={mse_ridge:.6f} | R^2={r2_ridge:.3f}")

# -------------------------
# 6) Coefficients (interpretable)
#    Note: after one-hot encoding, there are extra Hand dummy columns.
# -------------------------
# Get feature names after preprocessing
pre = linreg.named_steps["preprocess"]
num_names = NUM_FEATS
cat_names = list(pre.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CAT_FEATS))
all_feature_names = num_names + cat_names

coefs = linreg.named_steps["model"].coef_
coef_table = pd.DataFrame({"feature": all_feature_names, "coef": coefs})
coef_table["abs_coef"] = np.abs(coef_table["coef"])
coef_table = coef_table.sort_values("abs_coef", ascending=False).drop(columns="abs_coef")

print("\nTop coefficients (standardized numeric features; Hand is one-hot):")
print(coef_table.head(20).to_string(index=False))

# -------------------------
# 7) Plots
# -------------------------
# (a) Predicted vs True
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test)
mn = min(y_test.min(), y_pred_test.min())
mx = max(y_test.max(), y_pred_test.max())
plt.plot([mn, mx], [mn, mx])
plt.xlabel("True nWBV")
plt.ylabel("Predicted nWBV")
plt.title("Predicted vs True nWBV (Test Set)")
plt.tight_layout()
plt.show()

# (b) Residual plot
resid = y_test - y_pred_test
plt.figure(figsize=(6, 5))
plt.scatter(y_pred_test, resid)
plt.axhline(0.0)
plt.xlabel("Predicted nWBV")
plt.ylabel("Residual (True - Pred)")
plt.title("Residual Plot (Test Set)")
plt.tight_layout()
plt.show()

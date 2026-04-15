# ============================================================
# TASK 2 - House Price Prediction using Machine Learning
# CodTech Internship - Mallavarapu Venkata Sai (CTIS1591)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────
DATA_PATH    = "../data/train.csv"
TEST_PATH    = "../data/test.csv"
OUTPUTS_DIR  = "../outputs/"
MODELS_DIR   = "../models/"
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

print("=" * 60)
print("   TASK 2 - HOUSE PRICE PREDICTION")
print("=" * 60)

# ── 1. Load Data ─────────────────────────────────────────────
print("\n📂 Loading data...")
train = pd.read_csv(DATA_PATH)
test  = pd.read_csv(TEST_PATH)
print(f"   Train shape : {train.shape}")
print(f"   Test shape  : {test.shape}")

# ── 2. EDA ───────────────────────────────────────────────────
print("\n📊 Generating EDA plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Sale Price Distribution", fontsize=14, fontweight='bold')

sns.histplot(train["SalePrice"], kde=True, color="steelblue", ax=axes[0])
axes[0].set_title("Original")
axes[0].set_xlabel("Sale Price ($)")

sns.histplot(np.log1p(train["SalePrice"]), kde=True, color="green", ax=axes[1])
axes[1].set_title("Log Transformed")
axes[1].set_xlabel("Log Sale Price")

plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}01_price_distribution.png", dpi=150)
plt.close()
print("   ✅ price_distribution.png")

# ── 3. Missing Values ────────────────────────────────────────
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False).head(20)

plt.figure(figsize=(12, 6))
missing.plot(kind="bar", color="tomato")
plt.title("Top 20 Columns with Missing Values", fontweight='bold')
plt.ylabel("Missing Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}02_missing_values.png", dpi=150)
plt.close()
print("   ✅ missing_values.png")

# ── 4. Correlation Heatmap ───────────────────────────────────
num_cols = train.select_dtypes(include=[np.number]).columns
corr     = train[num_cols].corr()
top_corr = corr["SalePrice"].abs().sort_values(ascending=False).head(15).index

plt.figure(figsize=(13, 9))
sns.heatmap(train[top_corr].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5)
plt.title("Top 15 Feature Correlations", fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}03_correlation_heatmap.png", dpi=150)
plt.close()
print("   ✅ correlation_heatmap.png")

# ── 5. Top Correlated Features vs SalePrice ─────────────────
top_features = corr["SalePrice"].abs() \
    .sort_values(ascending=False)[1:7].index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Top Features vs Sale Price", fontsize=14, fontweight='bold')
for i, feat in enumerate(top_features):
    ax = axes[i//3][i%3]
    ax.scatter(train[feat], train["SalePrice"], alpha=0.3, color="purple")
    ax.set_xlabel(feat)
    ax.set_ylabel("SalePrice")
    ax.set_title(feat)
plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}04_feature_vs_price.png", dpi=150)
plt.close()
print("   ✅ feature_vs_price.png")

# ── 6. Preprocessing ─────────────────────────────────────────
print("\n⚙️  Preprocessing data...")

def preprocess(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    cat_cols = df.select_dtypes(include=["object"]).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
        df[col] = le.fit_transform(df[col].astype(str))
    return df

train_p = preprocess(train)
test_p  = preprocess(test)

features = [c for c in train_p.columns if c not in ["SalePrice","Id"]]
X = train_p[features]
y = np.log1p(train_p["SalePrice"])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"   Train : {X_train.shape}  |  Val : {X_val.shape}")

# ── 7. Train Models ──────────────────────────────────────────
print("\n🤖 Training models...")

models = {
    "Linear Regression"  : LinearRegression(),
    "Ridge Regression"   : Ridge(alpha=10),
    "Random Forest"      : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting"  : GradientBoostingRegressor(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    results[name] = {
        "RMSE" : round(np.sqrt(mean_squared_error(y_val, preds)), 4),
        "R2"   : round(r2_score(y_val, preds), 4),
        "MAE"  : round(mean_absolute_error(y_val, preds), 4),
    }
    print(f"   {name:<25} RMSE={results[name]['RMSE']}  "
          f"R2={results[name]['R2']}  MAE={results[name]['MAE']}")

# ── 8. Model Comparison Plot ─────────────────────────────────
results_df = pd.DataFrame(results).T
colors = ["#3498db","#e67e22","#2ecc71","#e74c3c"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Model Comparison", fontsize=14, fontweight='bold')

for ax, metric, title in zip(axes,
                              ["R2","RMSE","MAE"],
                              ["R² Score (higher=better)",
                               "RMSE (lower=better)",
                               "MAE (lower=better)"]):
    results_df[metric].plot(kind="bar", ax=ax, color=colors)
    ax.set_title(title)
    ax.set_xticklabels(results_df.index, rotation=30, ha='right')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}05_model_comparison.png", dpi=150)
plt.close()
print("\n   ✅ model_comparison.png")

# ── 9. Best Model ────────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]["R2"])
best_model = models[best_name]
best_preds = best_model.predict(X_val)
print(f"\n🏆 Best Model: {best_name}  (R²={results[best_name]['R2']})")

# ── 10. Feature Importance ───────────────────────────────────
if hasattr(best_model, "feature_importances_"):
    feat_imp = pd.Series(best_model.feature_importances_,
                         index=features).sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    feat_imp.plot(kind="barh", color="darkorange")
    plt.title(f"Top 15 Feature Importances ({best_name})", fontweight='bold')
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}06_feature_importance.png", dpi=150)
    plt.close()
    print("   ✅ feature_importance.png")

# ── 11. Actual vs Predicted ──────────────────────────────────
plt.figure(figsize=(8, 6))
plt.scatter(y_val, best_preds, alpha=0.4, color="purple", s=20)
plt.plot([y_val.min(), y_val.max()],
         [y_val.min(), y_val.max()], 'r--', lw=2, label="Perfect Fit")
plt.xlabel("Actual Log Price")
plt.ylabel("Predicted Log Price")
plt.title(f"Actual vs Predicted ({best_name})", fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}07_actual_vs_predicted.png", dpi=150)
plt.close()
print("   ✅ actual_vs_predicted.png")

# ── 12. Residuals Plot ───────────────────────────────────────
residuals = y_val - best_preds
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(best_preds, residuals, alpha=0.4, color="teal", s=20)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")

plt.subplot(1, 2, 2)
sns.histplot(residuals, kde=True, color="teal")
plt.xlabel("Residual")
plt.title("Residual Distribution")
plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}08_residuals.png", dpi=150)
plt.close()
print("   ✅ residuals.png")

# ── 13. Save Model & Report ──────────────────────────────────
print("\n💾 Saving model and report...")
joblib.dump(best_model, f"{MODELS_DIR}trained_model.pkl")
joblib.dump(models,     f"{MODELS_DIR}all_models.pkl")

report = pd.DataFrame(results).T
report.to_csv(f"{OUTPUTS_DIR}evaluation_report.csv")

print(f"   ✅ trained_model.pkl saved")
print(f"   ✅ evaluation_report.csv saved")

# ── 14. Final Summary ────────────────────────────────────────
print("\n" + "=" * 60)
print("   FINAL RESULTS SUMMARY")
print("=" * 60)
print(report.to_string())
print(f"\n🏆 Best Model  : {best_name}")
print(f"   R² Score    : {results[best_name]['R2']}")
print(f"   RMSE        : {results[best_name]['RMSE']}")
print(f"   MAE         : {results[best_name]['MAE']}")
print("\n✅ Task 2 Complete! All outputs saved in /outputs folder.")
print("=" * 60)
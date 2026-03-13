# ============================================================
# train_model.py — Hospital Readmission Prediction
# MSc IT Mini Project
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)
import joblib

# ── Output folder for plots ────────────────────────────────
os.makedirs('static/plots', exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("  Hospital Patient Readmission — ML Mini Project")
print("=" * 60)

df = pd.read_csv('dataset.csv')
print(f"\n✔  Dataset loaded  →  {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 3 rows:")
print(df.head(3).to_string())

# ============================================================
# 2. DATA PREPROCESSING
# ============================================================
print("\n--- 2. DATA PREPROCESSING ---")

# 2a. Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 2b. Replace 'Missing' strings with NaN and fill with mode
df.replace('Missing', np.nan, inplace=True)
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values after cleaning:", df.isnull().sum().sum())

# 2c. Encode categorical columns with LabelEncoder
le_dict = {}     # we'll save these encoders so we can use them in predict.py
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('readmitted')   # target — handle separately

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# Encode target
le_target = LabelEncoder()
df['readmitted'] = le_target.fit_transform(df['readmitted'])   # no=0, yes=1
print(f"\nTarget classes: {le_target.classes_}")   # [no, yes]

# Save encoders
joblib.dump(le_dict,    'le_dict.pkl')
joblib.dump(le_target,  'le_target.pkl')
print("✔  Label encoders saved.")

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS  (EDA)
# ============================================================
print("\n--- 3. EDA — generating plots ---")

sns.set_theme(style='whitegrid', palette='muted')

# Plot 1 — Target distribution
plt.figure(figsize=(5, 4))
ax = sns.countplot(x='readmitted', data=df, palette=['#4A90D9', '#27AE60'])
ax.set_xticklabels(['Not Readmitted (0)', 'Readmitted (1)'])
plt.title('Readmission Distribution', fontsize=13, fontweight='bold')
plt.xlabel(''); plt.ylabel('Count')
plt.tight_layout()
plt.savefig('static/plots/target_distribution.png', dpi=100)
plt.close()

# Plot 2 — Age group vs Readmission
plt.figure(figsize=(7, 4))
sns.countplot(x='age', hue='readmitted', data=df,
              palette=['#4A90D9', '#27AE60'])
plt.title('Age Group vs Readmission', fontsize=13, fontweight='bold')
plt.xlabel('Age (encoded)'); plt.ylabel('Count')
plt.legend(title='Readmitted', labels=['No', 'Yes'])
plt.tight_layout()
plt.savefig('static/plots/age_readmission.png', dpi=100)
plt.close()

# Plot 3 — Correlation heatmap (numeric cols)
plt.figure(figsize=(10, 7))
num_cols = df.select_dtypes(include=np.number).columns
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues',
            linewidths=0.5, annot_kws={'size': 8})
plt.title('Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('static/plots/correlation_heatmap.png', dpi=100)
plt.close()

# Plot 4 — Distribution of time_in_hospital
plt.figure(figsize=(6, 4))
sns.histplot(df['time_in_hospital'], bins=14, kde=True, color='#4A90D9')
plt.title('Distribution of Time in Hospital', fontsize=13, fontweight='bold')
plt.xlabel('Days'); plt.ylabel('Count')
plt.tight_layout()
plt.savefig('static/plots/time_in_hospital_dist.png', dpi=100)
plt.close()

# Plot 5 — n_medications distribution by readmission
plt.figure(figsize=(7, 4))
sns.boxplot(x='readmitted', y='n_medications', data=df,
            palette=['#4A90D9', '#27AE60'])
plt.xticks([0, 1], ['Not Readmitted', 'Readmitted'])
plt.title('Medications vs Readmission', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('static/plots/medications_readmission.png', dpi=100)
plt.close()

print("✔  5 plots saved to static/plots/")

# ============================================================
# 4. FEATURE SELECTION & TRAIN-TEST SPLIT
# ============================================================
print("\n--- 4. TRAIN-TEST SPLIT ---")

X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Save feature column order (needed for prediction)
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

# ============================================================
# 5. TRAIN MULTIPLE MODELS
# ============================================================
print("\n--- 5. TRAINING MODELS ---")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree'      : DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'KNN'                : KNeighborsClassifier(n_neighbors=7)
}

results = {}   # store accuracy for comparison

for name, model in models.items():
    print(f"\n  Training {name} …", end=' ')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {'model': model, 'accuracy': acc, 'y_pred': y_pred}
    print(f"Accuracy = {acc:.4f}")

# ============================================================
# 6. EVALUATE & COMPARE MODELS
# ============================================================
print("\n--- 6. MODEL EVALUATION ---")

for name, res in results.items():
    print(f"\n{'='*50}")
    print(f"  {name}  —  Accuracy: {res['accuracy']:.4f}")
    print(f"{'='*50}")
    cm = confusion_matrix(y_test, res['y_pred'])
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:")
    print(classification_report(y_test, res['y_pred'],
                                target_names=le_target.classes_))

# Best model
best_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_name]['model']
print(f"\n🏆  Best Model : {best_name}  "
      f"(Accuracy = {results[best_name]['accuracy']:.4f})")

# Accuracy comparison bar chart
plt.figure(figsize=(8, 4))
names = list(results.keys())
accs  = [results[n]['accuracy'] for n in names]
colors = ['#27AE60' if n == best_name else '#4A90D9' for n in names]
bars = plt.bar(names, accs, color=colors, edgecolor='white', linewidth=1.2)
plt.ylim(0.5, 1.0)
plt.title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
plt.ylabel('Accuracy')
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{acc:.3f}', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('static/plots/model_comparison.png', dpi=100)
plt.close()
print("✔  Model comparison chart saved.")

# ============================================================
# 7. SAVE BEST MODEL
# ============================================================
joblib.dump(best_model, 'model.pkl')
print(f"\n✔  Best model ({best_name}) saved as model.pkl")

print("\n" + "="*60)
print("  Training complete! Run app.py to start the web interface.")
print("="*60)

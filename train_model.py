"""
Train Random Forest model and generate all graphs.
Run this once before starting the Flask app: python train_model.py
"""
import os
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              roc_curve, auc, classification_report)

warnings.filterwarnings('ignore')

GRAPHS = 'static/graphs'
os.makedirs(GRAPHS, exist_ok=True)
os.makedirs('model', exist_ok=True)

# ── Color palette ──────────────────────────────────────────────────
COLORS = ['#00B4D8','#0077B6','#48CAE4','#ADE8F4','#90E0EF',
          '#023E8A','#03045E','#CAF0F8']
ACCENT = '#0077B6'

# ── Load & preprocess ───────────────────────────────────────────────
print("[1/4] Loading data...")
df = pd.read_csv('static/data/hospital_data_2200.csv')

num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# IQR outlier capping
for col in num_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

# Store encoders
encoders = {}
for col in cat_cols:
    if col != 'readmitted':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

le_target = LabelEncoder()
df['readmitted'] = le_target.fit_transform(df['readmitted'])
encoders['readmitted'] = le_target

X = df.drop('readmitted', axis=1)
y = df['readmitted']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── DATA GRAPHS ─────────────────────────────────────────────────────
print("[2/4] Generating data graphs...")

def savefig(name):
    plt.tight_layout()
    plt.savefig(f'{GRAPHS}/{name}', dpi=110, bbox_inches='tight',
                facecolor='#f0f8ff')
    plt.close()

# 1. Yes/No column bar chart
yes_no_cols = {}
original_df = pd.read_csv('static/data/hospital_data_2200.csv')
for col in original_df.select_dtypes(include=['object']).columns:
    vals = original_df[col].dropna().unique()
    if set([str(v).lower() for v in vals]).issuperset({'yes','no'}):
        yes_no_cols[col] = original_df[col].value_counts()

fig, ax = plt.subplots(figsize=(10, 5), facecolor='#f0f8ff')
ax.set_facecolor('#e8f4fd')
if yes_no_cols:
    col_names = list(yes_no_cols.keys())
    yes_counts = [yes_no_cols[c].get('yes', 0) for c in col_names]
    no_counts  = [yes_no_cols[c].get('no', 0)  for c in col_names]
    x = np.arange(len(col_names))
    w = 0.35
    ax.bar(x - w/2, yes_counts, w, label='Yes', color='#00B4D8', edgecolor='white', linewidth=1.5)
    ax.bar(x + w/2, no_counts,  w, label='No',  color='#0077B6', edgecolor='white', linewidth=1.5)
    ax.set_xticks(x); ax.set_xticklabels(col_names, rotation=20, ha='right', fontsize=11)
ax.set_title('Yes / No Column Distribution', fontsize=16, fontweight='bold', color='#023E8A', pad=12)
ax.set_ylabel('Count', fontsize=12, color='#023E8A')
ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.4)
savefig('yes_no_bar.png')

# 2. Age Distribution
fig, ax = plt.subplots(figsize=(10,5), facecolor='#f0f8ff')
ax.set_facecolor('#e8f4fd')
age_counts = original_df['age'].value_counts().sort_index()
bars = ax.bar(age_counts.index, age_counts.values,
              color=COLORS[:len(age_counts)], edgecolor='white', linewidth=1.5)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+5,
            str(int(bar.get_height())), ha='center', va='bottom', fontsize=10, color='#023E8A')
ax.set_title('Age Distribution', fontsize=16, fontweight='bold', color='#023E8A', pad=12)
ax.set_xlabel('Age Group', fontsize=12, color='#023E8A')
ax.set_ylabel('Count', fontsize=12, color='#023E8A')
ax.grid(axis='y', alpha=0.4)
savefig('age_distribution.png')

# 3. Data Overview – readmitted distribution
fig, axes = plt.subplots(1, 2, figsize=(12,5), facecolor='#f0f8ff')
for a in axes: a.set_facecolor('#e8f4fd')
rc = original_df['readmitted'].value_counts()
axes[0].bar(rc.index, rc.values, color=['#00B4D8','#0077B6'], edgecolor='white', linewidth=2)
axes[0].set_title('Readmitted Distribution', fontsize=14, fontweight='bold', color='#023E8A')
axes[0].set_ylabel('Count', color='#023E8A')
axes[0].grid(axis='y', alpha=0.4)
axes[1].pie(rc.values, labels=rc.index, autopct='%1.1f%%',
            colors=['#00B4D8','#0077B6'], startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':2})
axes[1].set_title('Readmitted Proportion', fontsize=14, fontweight='bold', color='#023E8A')
savefig('data_overview.png')

# 4. Heatmap
fig, ax = plt.subplots(figsize=(12,9), facecolor='#f0f8ff')
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', ax=ax,
            cmap='Blues', linewidths=0.5, annot_kws={'size':8},
            cbar_kws={'shrink':0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', color='#023E8A', pad=12)
savefig('heatmap.png')

# 5. Numerical feature distributions
num_features = [c for c in df.columns if c != 'readmitted' and df[c].dtype in ['float64','int64']]
n = len(num_features)
cols_grid = 3
rows_grid = (n + cols_grid - 1) // cols_grid
fig, axes = plt.subplots(rows_grid, cols_grid, figsize=(14, rows_grid*3.5), facecolor='#f0f8ff')
axes = axes.flatten()
for i, col in enumerate(num_features):
    axes[i].set_facecolor('#e8f4fd')
    axes[i].hist(df[col], bins=25, color=COLORS[i % len(COLORS)], edgecolor='white', linewidth=0.7)
    axes[i].set_title(col, fontsize=11, fontweight='bold', color='#023E8A')
    axes[i].grid(axis='y', alpha=0.3)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold', color='#023E8A', y=1.01)
savefig('feature_distributions.png')

# ── MODEL GRAPHS ────────────────────────────────────────────────────

# ── GridSearchCV — Random Forest only ───────────────────────────────
print("[3/4] Running GridSearchCV on Random Forest (this may take a minute)...")
rf_param_grid = {
    'n_estimators':      [300],
    'max_depth':         [20],
    'min_samples_split': [2],
    'min_samples_leaf':  [1],
    'max_features':      ['sqrt'],
}
rf_base = RandomForestClassifier(bootstrap=True, random_state=42)
rf_grid_search = GridSearchCV(
    rf_base,
    rf_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
rf_grid_search.fit(X_train, y_train)

print(f"   Best RF params  : {rf_grid_search.best_params_}")
print(f"   Best RF CV score: {rf_grid_search.best_score_:.4f}")
best_rf = rf_grid_search.best_estimator_

# Train all models (CV=5)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Random Forest':       best_rf,
    'SVM':                 SVC(probability=True, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
    'AdaBoost':            AdaBoostClassifier(random_state=42),
    'Naive Bayes':         GaussianNB(),
}

cv_scores, train_accs, test_accs = {}, {}, {}
fitted = {}

for name, model in models.items():
    print(f"   Training {name}...")
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores[name] = cv.mean()
    model.fit(X_train, y_train)
    train_accs[name] = accuracy_score(y_train, model.predict(X_train))
    test_accs[name]  = accuracy_score(y_test,  model.predict(X_test))
    fitted[name] = model

rf_model = fitted['Random Forest']

# 6. Model Accuracy Bar Chart
fig, ax = plt.subplots(figsize=(12,6), facecolor='#f0f8ff')
ax.set_facecolor('#e8f4fd')
names = list(cv_scores.keys())
scores = [cv_scores[n] for n in names]
bar_colors = COLORS[:len(names)]
bars = ax.bar(names, scores, color=bar_colors, edgecolor='white', linewidth=1.5)
for bar, s in zip(bars, scores):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f'{s:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#023E8A')
ax.set_ylim(0, 1.05)
ax.set_ylabel('CV Accuracy (5-fold)', fontsize=12, color='#023E8A')
ax.set_title('Model Accuracy Comparison (CV=5)', fontsize=16, fontweight='bold', color='#023E8A', pad=12)
ax.tick_params(axis='x', rotation=20)
ax.grid(axis='y', alpha=0.4)
savefig('model_accuracy_bar.png')

# 7. ROC Curve (all models)
fig, ax = plt.subplots(figsize=(10,7), facecolor='#f0f8ff')
ax.set_facecolor('#e8f4fd')
roc_colors = ['#00B4D8','#0077B6','#FF6B35','#48CAE4','#023E8A','#7B2D8B','#2ECC71']
for (name, model), color in zip(fitted.items(), roc_colors):
    probs = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    lw = 3 if name == 'Random Forest' else 1.5
    ax.plot(fpr, tpr, lw=lw, color=color, label=f'{name} (AUC={roc_auc:.3f})')
ax.plot([0,1],[0,1],'k--',lw=1.2,alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12, color='#023E8A')
ax.set_ylabel('True Positive Rate', fontsize=12, color='#023E8A')
ax.set_title('ROC Curves — All Models', fontsize=16, fontweight='bold', color='#023E8A', pad=12)
ax.legend(fontsize=9, loc='lower right')
ax.grid(alpha=0.3)
savefig('roc_curve.png')

# 8. Confusion Matrix (Random Forest)
fig, ax = plt.subplots(figsize=(7,6), facecolor='#f0f8ff')
cm = confusion_matrix(y_test, rf_model.predict(X_test))
labels = le_target.classes_
sns.heatmap(cm, annot=True, fmt='d', ax=ax,
            cmap='Blues', xticklabels=labels, yticklabels=labels,
            linewidths=2, linecolor='white',
            annot_kws={'size':14,'fontweight':'bold'})
ax.set_xlabel('Predicted', fontsize=13, color='#023E8A')
ax.set_ylabel('Actual', fontsize=13, color='#023E8A')
ax.set_title('Confusion Matrix — Random Forest', fontsize=16, fontweight='bold', color='#023E8A', pad=12)
savefig('confusion_matrix.png')

# 9. Feature Importance (Random Forest)
fig, ax = plt.subplots(figsize=(10,7), facecolor='#f0f8ff')
ax.set_facecolor('#e8f4fd')
importances = rf_model.feature_importances_
idx = np.argsort(importances)[::-1]
sorted_feats = [feature_names[i] for i in idx]
sorted_imps  = importances[idx]
bar_c = [COLORS[i % len(COLORS)] for i in range(len(sorted_feats))]
bars = ax.barh(sorted_feats[::-1], sorted_imps[::-1], color=bar_c[::-1], edgecolor='white', linewidth=1)
ax.set_xlabel('Importance Score', fontsize=12, color='#023E8A')
ax.set_title('Feature Importance — Random Forest', fontsize=16, fontweight='bold', color='#023E8A', pad=12)
ax.grid(axis='x', alpha=0.4)
savefig('feature_importance.png')

# 10. Overfitting / Underfitting Graph
fig, ax = plt.subplots(figsize=(12,6), facecolor='#f0f8ff')
ax.set_facecolor('#e8f4fd')
x = np.arange(len(names))
w = 0.28
ax.bar(x - w, [train_accs[n] for n in names], w, label='Train Acc', color='#00B4D8', edgecolor='white')
ax.bar(x,     [test_accs[n]  for n in names], w, label='Test Acc',  color='#0077B6', edgecolor='white')
ax.bar(x + w, [cv_scores[n]  for n in names], w, label='CV Acc',    color='#023E8A', edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
ax.set_ylim(0, 1.1)

ax.set_ylabel('Accuracy', fontsize=12, color='#023E8A')
ax.set_title('Overfitting / Underfitting Analysis', fontsize=16, fontweight='bold', color='#023E8A', pad=12)
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.4)
savefig('overfitting_graph.png')

# 11. Cross-val score distribution (Random Forest)
fig, ax = plt.subplots(figsize=(9,5), facecolor='#f0f8ff')
ax.set_facecolor('#e8f4fd')
rf_cv = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
folds = [f'Fold {i+1}' for i in range(5)]
bars = ax.bar(folds, rf_cv, color=COLORS[:5], edgecolor='white', linewidth=2)
for bar, s in zip(bars, rf_cv):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f'{s:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#023E8A')
ax.axhline(rf_cv.mean(), color='#FF6B35', linestyle='--', linewidth=2, label=f'Mean={rf_cv.mean():.3f}')
ax.set_ylim(0,1.1)
ax.set_ylabel('Accuracy', fontsize=12, color='#023E8A')
ax.set_title('Random Forest — CV=5 Fold Scores', fontsize=16, fontweight='bold', color='#023E8A', pad=12)
ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.4)
savefig('cv_scores.png')

# ── Save model & artifacts ──────────────────────────────────────────
print("[4/4] Saving model and encoders...")
with open('model/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('model/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
with open('model/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print("\n✅ Training complete!")
print(f"   Random Forest Test Accuracy : {test_accs['Random Forest']:.4f}")
print(f"   Random Forest CV Accuracy   : {cv_scores['Random Forest']:.4f}")
print(f"   GridSearchCV Best Params    : {rf_grid_search.best_params_}")
print("   All graphs saved to static/graphs/")
print("   Model saved to model/")

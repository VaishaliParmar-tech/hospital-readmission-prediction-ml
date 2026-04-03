# 🏥 MediPredict — Hospital Readmission Prediction

A complete Flask + Machine Learning web application that predicts hospital readmission
for diabetic patients using a Random Forest classifier.

---

## 🚀 Quick Start (3 Steps)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the model & generate graphs
```bash
python train_model.py
```
> ⏳ This trains 7 ML models with CV=5. Takes 2–5 minutes. Only needs to run once.

### Step 3 — Start the web app
```bash
python app.py
```
Then open your browser at: **http://localhost:5000**

---

## 📁 Project Structure
```
hospital_project/
├── app.py                  # Flask web application
├── train_model.py          # Model training + graph generation
├── requirements.txt        # Python dependencies
├── model/                  # Saved model files (created after training)
│   ├── rf_model.pkl
│   ├── encoders.pkl
│   └── feature_names.pkl
├── static/
│   ├── css/style.css       # All styles
│   ├── data/
│   │   └── hospital_data_2200.csv  # Dataset
│   └── graphs/             # Generated graphs (created after training)
└── templates/
    ├── home.html
    ├── prediction.html
    └── about.html
```

---

## 📊 Pages
- **Home** — Project intro, features, methodology overview
- **Prediction** — Patient form + Data Graphs + Model Graphs + Prediction result
- **About** — Team info, technologies, methodology details

## 🌳 ML Models (CV=5)
| Model | Role |
|---|---|
| **Random Forest** | ⭐ PRIMARY MODEL |
| Logistic Regression | Comparison |
| Decision Tree | Comparison |
| SVM | Comparison |
| KNN | Comparison |
| AdaBoost | Comparison |
| Naive Bayes | Comparison |

## 📈 Graphs Generated
### Data Graphs
- Dataset Overview (Readmission Distribution)
- Age Distribution
- Yes/No Column Bar Chart
- Feature Correlation Heatmap
- All Feature Distributions

### Model Graphs
- Model Accuracy Comparison Bar Chart
- ROC Curves (All Models)
- Confusion Matrix (Random Forest)
- Feature Importance (Random Forest)
- Overfitting/Underfitting Analysis
- CV=5 Fold Scores (Random Forest)

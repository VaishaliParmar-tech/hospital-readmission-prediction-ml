"""
Hospital Readmission Prediction - Flask Application
Run: python app.py
"""
import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.secret_key = 'hospital_ml_secret_2024'

MODEL_DIR = 'model'

# Global variables
rf_model = None
encoders = None
feature_names = None
MODEL_READY = False

def load_artifacts():
    global rf_model, encoders, feature_names, MODEL_READY
    try:
        with open(os.path.join(MODEL_DIR, 'rf_model.pkl'), 'rb') as f:
            rf_model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'encoders.pkl'), 'rb') as f:
            encoders = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'rb') as f:
            feature_names = pickle.load(f)
        MODEL_READY = True
        print("[OK] Model loaded successfully.")
    except Exception as e:
        MODEL_READY = False
        print("[WARNING] Model not loaded: " + str(e))
        print("  Run 'python train_model.py' first.")

load_artifacts()


def get_choices():
    """Return dropdown choices as a plain dict. Always returns a dict, never a tuple."""
    empty = {
        'age_groups':    [],
        'med_specialty': [],
        'diag_options':  [],
        'glucose_opts':  [],
        'a1c_opts':      [],
        'change_opts':   [],
        'diabetes_opts': [],
    }
    if encoders is None:
        return empty
    try:
        return {
            'age_groups':    list(encoders['age'].classes_),
            'med_specialty': list(encoders['medical_specialty'].classes_),
            'diag_options':  list(encoders['diag_1'].classes_),
            'glucose_opts':  list(encoders['glucose_test'].classes_),
            'a1c_opts':      list(encoders['A1Ctest'].classes_),
            'change_opts':   list(encoders['change'].classes_),
            'diabetes_opts': list(encoders['diabetes'].classes_),
        }
    except Exception:
        return empty


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/prediction')
def prediction():
    choices = get_choices()
    return render_template(
        'prediction.html',
        model_ready=MODEL_READY,
        age_groups=choices['age_groups'],
        med_specialty=choices['med_specialty'],
        diag_options=choices['diag_options'],
        glucose_opts=choices['glucose_opts'],
        a1c_opts=choices['a1c_opts'],
        change_opts=choices['change_opts'],
        diabetes_opts=choices['diabetes_opts'],
    )


@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_READY:
        return jsonify({'error': 'Model not ready. Run python train_model.py first.'}), 500

    try:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({'error': 'Invalid JSON payload.'}), 400

        row = {}

        # Numerical fields
        num_fields = [
            'time_in_hospital', 'n_lab_procedures', 'n_procedures',
            'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency'
        ]
        for key in num_fields:
            try:
                row[key] = float(data.get(key, 0))
            except (ValueError, TypeError):
                row[key] = 0.0

        # Categorical fields - encode with saved LabelEncoders
        cat_fields = [
            'age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
            'glucose_test', 'A1Ctest', 'change', 'diabetes'
        ]
        for key in cat_fields:
            val = str(data.get(key, ''))
            enc = encoders.get(key)
            if enc is not None and val in list(enc.classes_):
                row[key] = int(enc.transform([val])[0])
            else:
                row[key] = 0

        # Build DataFrame with correct column order
        X_input = pd.DataFrame(
            [[row[f] for f in feature_names]],
            columns=feature_names
        )

        pred  = rf_model.predict(X_input)[0]
        proba = rf_model.predict_proba(X_input)[0]

        le_target  = encoders['readmitted']
        pred_label = str(le_target.inverse_transform([pred])[0])
        classes    = list(le_target.classes_)
        prob_yes   = float(proba[classes.index('yes')]) if 'yes' in classes else float(proba[1])

        return jsonify({
            'prediction':  pred_label,
            'probability': round(prob_yes * 100, 2),
            'risk_level':  'High Risk' if pred_label == 'yes' else 'Low Risk'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

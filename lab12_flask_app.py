# Asad | BSAI 4C | Roll: 141
# Lab 12: Flask Application for Titanic Survival Prediction

from flask import Flask, request, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and column order
script_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(script_dir, 'titanic_model.pkl'))
columns = joblib.load(os.path.join(script_dir, 'model_columns.pkl'))

# HTML form (frontend code generated with ChatGPT help)
HTML_FORM = '''
<!DOCTYPE html>
<html>
<head>
    <title>Titanic Survival Predictor</title>
    <style>
        body { font-family: Arial; background: #f0f8ff; text-align: center; padding: 30px; }
        form { display: inline-block; background: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 0 15px rgba(0,0,0,0.2); }
        label { display: block; margin-top: 10px; font-weight: bold; }
        input, select { width: 100%; padding: 8px; margin-top: 5px; border-radius: 5px; border: 1px solid #ccc; }
        button { background: #008080; color: white; border: none; padding: 10px 20px; margin-top: 20px; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background: #005f5f; }
        .result { margin-top: 20px; font-size: 20px; }
    </style>
</head>
<body>
    <h1>Titanic Survival Predictor</h1>
    <form action="/predict" method="post">
        <label>Passenger Class (1,2,3):</label>
        <input type="number" name="Pclass" min="1" max="3" required>

        <label>Sex (0 = male, 1 = female):</label>
        <select name="Sex" required>
            <option value="0">Male</option>
            <option value="1">Female</option>
        </select>

        <label>Age:</label>
        <input type="number" step="any" name="Age" placeholder="e.g., 25" required>

        <label>SibSp (siblings/spouses aboard):</label>
        <input type="number" name="SibSp" value="0" required>

        <label>Parch (parents/children aboard):</label>
        <input type="number" name="Parch" value="0" required>

        <label>Fare:</label>
        <input type="number" step="any" name="Fare" placeholder="e.g., 32.2" required>

        <label>Embarked (0=S, 1=C, 2=Q):</label>
        <select name="Embarked" required>
            <option value="0">Southampton (S)</option>
            <option value="1">Cherbourg (C)</option>
            <option value="2">Queenstown (Q)</option>
        </select>

        <button type="submit">Predict Survival</button>
    </form>

    {% if prediction is not none %}
        <div class="result">
            {% if prediction == 1 %}
                <p style="color: green;">Survived</p>
            {% else %}
                <p style="color: red;">Did not survive</p>
            {% endif %}
        </div>
    {% endif %}
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_FORM, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data and build input array in correct column order
    input_dict = {
        'PassengerId': 0,          # dummy, not used by model but column exists in training
        'Pclass': int(request.form['Pclass']),
        'Name': 0,                 # encoded, same dummy
        'Sex': int(request.form['Sex']),
        'Age': float(request.form['Age']),
        'SibSp': int(request.form['SibSp']),
        'Parch': int(request.form['Parch']),
        'Ticket': 0,               # dummy
        'Fare': float(request.form['Fare']),
        'Embarked': int(request.form['Embarked'])
    }

    # Ensure the order matches the training columns
    input_list = [input_dict[col] for col in columns]
    input_array = np.array([input_list])

    # Make prediction
    pred = model.predict(input_array)[0]

    return render_template_string(HTML_FORM, prediction=int(pred))

if __name__ == '__main__':
    app.run(debug=True)

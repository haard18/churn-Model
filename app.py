import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, jsonify, render_template
import pickle
from flask_cors import CORS

app = Flask("__name__")
CORS(app)
df_1 = pd.read_csv("first_telc.csv")

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/predict", methods=['POST'])
def predict():
    # Parse JSON data from request
    data = request.get_json()
    
    # Extract individual features from JSON data
    SeniorCitizen = data.get('SeniorCitizen')
    MonthlyCharges = data.get('MonthlyCharges')
    TotalCharges = data.get('TotalCharges')
    gender = data.get('gender')
    Partner = data.get('Partner')
    Dependents = data.get('Dependents')
    PhoneService = data.get('PhoneService')
    MultipleLines= data.get('MultipleLines')
    InternetService= data.get('InternetService')
    OnlineSecurity= data.get('OnlineSecurity')
    OnlineBackup= data.get('OnlineBackup')
    DeviceProtection = data.get('DeviceProtection')
    TechSupport= data.get('TechSupport')
    StreamingTV = data.get('StreamingTV')
    StreamingMovies = data.get('StreamingMovies')
    Contract= data.get('Contract')
    PaperlessBilling = data.get('PaperlessBilling')
    PaymentMethod= data.get('PaymentMethod')
    tenure = data.get('tenure')

    model = pickle.load(open("model.sav", "rb"))
    
    data_list = [[SeniorCitizen, MonthlyCharges, TotalCharges, gender, Partner, Dependents, PhoneService, 
                  MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, 
                  StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, tenure]]
    new_df = pd.DataFrame(data_list, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                              'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                              'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                              'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                              'PaymentMethod', 'tenure'])
    
    df_2 = pd.concat([df_1, new_df], ignore_index=True)
    
    # Ensure the tenure column contains only valid integers
    df_2['tenure'] = pd.to_numeric(df_2['tenure'], errors='coerce')
    
    # Drop rows with invalid tenure values
    df_2.dropna(subset=['tenure'], inplace=True)
    
    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
    df_2.drop(columns=['tenure'], axis=1, inplace=True)
    
    new_df_dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                           'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                           'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])
    
    # Remove duplicate columns
    new_df_dummies = new_df_dummies.loc[:, ~new_df_dummies.columns.duplicated()]

    # Align columns with the training data
    final_df = new_df_dummies.reindex(columns=model.feature_names_in_, fill_value=0)
    
    single = model.predict(final_df.tail(1))
    probability = model.predict_proba(final_df.tail(1))[:, 1]
    
    if single == 1:
        result = {
            "prediction": "This customer is likely to be churned!!",
            "confidence": "{}%".format(probability[0] * 100)
        }
    else:
        result = {
            "prediction": "This customer is likely to continue!!",
            "confidence": "{}%".format(probability[0] * 100)
        }
        
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

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
    inputQuery1 = data.get('query1')
    inputQuery2 = data.get('query2')
    inputQuery3 = data.get('query3')
    inputQuery4 = data.get('query4')
    inputQuery5 = data.get('query5')
    inputQuery6 = data.get('query6')
    inputQuery7 = data.get('query7')
    inputQuery8 = data.get('query8')
    inputQuery9 = data.get('query9')
    inputQuery10 = data.get('query10')
    inputQuery11 = data.get('query11')
    inputQuery12 = data.get('query12')
    inputQuery13 = data.get('query13')
    inputQuery14 = data.get('query14')
    inputQuery15 = data.get('query15')
    inputQuery16 = data.get('query16')
    inputQuery17 = data.get('query17')
    inputQuery18 = data.get('query18')
    inputQuery19 = data.get('query19')

    model = pickle.load(open("model.sav", "rb"))
    
    data_list = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
                  inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
                  inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]
    
    new_df = pd.DataFrame(data_list, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                              'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                              'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                              'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                              'PaymentMethod', 'tenure'])
    
    df_2 = pd.concat([df_1, new_df], ignore_index=True)
    
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

from flask import Flask, request, jsonify
import numpy as np 
import pickle
from flask_cors import CORS
import pandas as pd
from sklearn.metrics import r2_score
import json





app = Flask(__name__)
CORS(app)
model_placement=pickle.load(open("placement.pkl","rb"))
with open("data_file.json","rb") as json_file:
    data=json.load(json_file)
    accuracy=data
    



# For the home route
@app.route('/')
def home():
    response = {
        'message': 'Welcome to the home route!'
    }
    return jsonify(response)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Example: Extracting a specific value from the data
    value1 = data.get('Cgpa')
    
    # model lai chaiyea jastei requirement meet gareko
    values = [float(value1)]
    features=[np.array(values)]

    # Model maa pathaalp

    prediction=model_placement.predict(features)


  
    
    # Aako data lai response maa pathaako 
    # But this modal is always giving the same answer response
    response = {
       "prediction": prediction.tolist(),
       "accuracy":accuracy
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)



# ###########################
# from flask import Flask, request, jsonify
# import numpy as np 
# import pandas as pd
# import pickle
# from flask_cors import CORS
# from sklearn.metrics import r2_score

# app = Flask(__name__)
# CORS(app)

# # Load the trained model and test data
# model_placement = pickle.load(open("placement.pkl", "rb"))
# y_test = pd.read_csv("output.csv")

# # For the home route
# @app.route('/')
# def home():
#     response = {
#         'message': 'Welcome to the home route!'
#     }
#     return jsonify(response)

# # Predict route
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     # Example: Extracting a specific value from the data
#     value1 = data.get('Cgpa')

#     # Prepare the features for prediction
#     features = np.array([float(value1)])

#     # Make predictions using the model
#     prediction = model_placement.predict(features.reshape(1, -1))

#     # Calculate accuracy
#     accuracy = r2_score(y_test, prediction)

#     # Prepare the response
#     response = {
#         "prediction": prediction.tolist(),
#         "accuracy": accuracy
#     }

#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)

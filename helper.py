import pandas as pd


def preprocess_input(data):

    df = pd.DataFrame({

        # Numerical Features
        "CreditScore": [data["CreditScore"]],

        # Geography Encoding
        "Geography_Germany": [
            1 if data["Geography"] == "Germany" else 0
        ],

        "Geography_Spain": [
            1 if data["Geography"] == "Spain" else 0
        ],

        # Gender Encoding
        "Gender_Male": [
            1 if data["Gender"] == "Male" else 0
        ],

        "Age": [data["Age"]],
        "Tenure": [data["Tenure"]],
        "Balance": [data["Balance"]],
        "NumOfProducts": [data["NumOfProducts"]],
        "HasCrCard": [data["HasCrCard"]],
        "IsActiveMember": [data["IsActiveMember"]],
        "EstimatedSalary": [data["EstimatedSalary"]]

    })

    return df
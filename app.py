import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.ensemble import StackingClassifier

app = Flask(__name__)

# Load the stacking ensemble model
with open('stacking_ensemble_model.pkl', 'rb') as model_file:
    stacking_model = joblib.load(model_file)

# Preprocessing functions
def fill_nans_by_age_and_cryosleep(df):
    df["RoomService"] = np.where((df["Age"] < 13) | (df["CryoSleep"] == True), 0, df["RoomService"])
    df["FoodCourt"] = np.where((df["Age"] < 13) | (df["CryoSleep"] == True), 0, df["FoodCourt"])
    df["ShoppingMall"] = np.where((df["Age"] < 13) | (df["CryoSleep"] == True), 0, df["ShoppingMall"])
    df["Spa"] = np.where((df["Age"] < 13) | (df["CryoSleep"] == True), 0, df["Spa"])
    df["VRDeck"] = np.where((df["Age"] < 13) | (df["CryoSleep"] == True), 0, df["VRDeck"])    
    return df

def clipping_quantile(dataframe, quantile_values=None, quantile=0.99):
    df = dataframe.copy()
    if quantile_values is None:
        quantile_values = df[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].quantile(quantile)
    for num_column in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
        num_values = df[num_column].values
        threshold = quantile_values[num_column]
        num_values = np.where(num_values > threshold, threshold, num_values)
        df[num_column] = num_values
    return df

def extract_features(df):
    df["PassengerGroup"] = (df["PassengerId"].str.split('_', expand=True))[0]
    No_People_In_PassengerGroup = df.groupby('PassengerGroup').aggregate({'PassengerId': 'size'}).reset_index()
    No_People_In_PassengerGroup = No_People_In_PassengerGroup.rename(columns={"PassengerId": "NoInPassengerGroup"})
    No_People_In_PassengerGroup["IsAlone"] = No_People_In_PassengerGroup["NoInPassengerGroup"].apply(lambda x: "Not Alone" if x > 1 else "Alone")
    df = df.merge(No_People_In_PassengerGroup[["PassengerGroup", "IsAlone"]], how='left', on=['PassengerGroup'])
    df["CabinDeck"] = df["Cabin"].str.split('/', expand=True)[0]
    df["DeckPosition"] = df["CabinDeck"].apply(lambda deck: "Lower" if deck in ('A', 'B', 'C', 'D') else "Higher")
    df["CabinSide"] = df["Cabin"].str.split('/', expand=True)[2]
    df["Regular"] = df["FoodCourt"] + df["ShoppingMall"]
    df["Luxury"] = df["RoomService"] + df["Spa"] + df["VRDeck"]
    df["TotalSpendings"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]
    Wealthiest_Deck = df.groupby('CabinDeck').aggregate({'TotalSpendings': 'sum', 'PassengerId': 'size'}).reset_index()
    Wealthiest_Deck['DeckAverageSpent'] = Wealthiest_Deck['TotalSpendings'] / Wealthiest_Deck['PassengerId']
    df = df.merge(Wealthiest_Deck[["CabinDeck", "DeckAverageSpent"]], how='left', on=['CabinDeck'])
    df["FamilyName"] = df["Name"].str.split(' ', expand=True)[1]
    NoRelatives = df.groupby('FamilyName')['PassengerId'].count().reset_index()
    NoRelatives = NoRelatives.rename(columns={"PassengerId": "NoRelatives"})
    df = df.merge(NoRelatives[["FamilyName", "NoRelatives"]], how='left', on=['FamilyName'])
    df["FamilySizeCat"] = pd.cut(df.NoRelatives, bins=[0, 2, 5, 10, 300], labels=['0 - 2', '3 - 5', '6 - 10', '11 - 208'])
    return df

def preprocess(df):
    # Handle missing categorical data
    list_missing_cat_columns = list((df.select_dtypes(['object', 'category']).isna().sum() > 0).index)
    for col in list_missing_cat_columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Handle missing numeric data
    list_missing_numeric_col = list((df.select_dtypes(np.number).isna().sum() > 0).index)
    df = fill_nans_by_age_and_cryosleep(df)
    for col in list_missing_numeric_col:
        df[col] = df[col].fillna(df[col].mean())

    df = clipping_quantile(df, None, 0.99)
    df = extract_features(df)

    # Drop irrelevant columns
    irrelevant_columns = ["Cabin", "PassengerId", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Name", "FamilyName", "PassengerGroup"]
    df = df.drop(irrelevant_columns, axis=1)

    # Encode categorical variables
    df = pd.get_dummies(df, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'CabinSide', 'IsAlone'])
    for col in ['CabinDeck', 'DeckPosition', 'FamilySizeCat']:
        df[col], _ = df[col].factorize()

    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Save the uploaded file temporarily
            temp_filepath = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(temp_filepath)

            # Read the CSV file
            df = pd.read_csv(temp_filepath)

            # Preprocess the data
            df_preprocessed = preprocess(df)

            # Make predictions
            predictions = stacking_model.predict(df_preprocessed)

            # Calculate percentages
            true_percentage = (predictions == 1).mean() * 100
            false_percentage = (predictions == 0).mean() * 100

            # Clean up: remove the temporary file
            os.remove(temp_filepath)

            return render_template('result.html', true_percentage=true_percentage, false_percentage=false_percentage)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)

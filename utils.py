import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

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
    df["PassengerGroup"] = df["PassengerId"].str.split('_', expand=True)[0]
    
    no_people = df.groupby('PassengerGroup').size().reset_index(name='NoInPassengerGroup')
    no_people["IsAlone"] = no_people["NoInPassengerGroup"].apply(lambda x: "Not Alone" if x > 1 else "Alone")
    df = df.merge(no_people, on='PassengerGroup', how='left')
    
    df["CabinDeck"] = df["Cabin"].str.split('/', expand=True)[0]
    df["DeckPosition"] = df["CabinDeck"].apply(lambda deck: "Lower" if deck in ('A', 'B', 'C', 'D') else "Higher")
    df["CabinSide"] = df["Cabin"].str.split('/', expand=True)[2]
    
    df["Regular"] = df["FoodCourt"] + df["ShoppingMall"]
    df["Luxury"] = df["RoomService"] + df["Spa"] + df["VRDeck"]
    df["TotalSpendings"] = df["Regular"] + df["Luxury"]
    
    wealthiest_deck = df.groupby('CabinDeck').agg({'TotalSpendings': 'sum', 'PassengerId': 'count'}).reset_index()
    wealthiest_deck['DeckAverageSpent'] = wealthiest_deck['TotalSpendings'] / wealthiest_deck['PassengerId']
    df = df.merge(wealthiest_deck[['CabinDeck', 'DeckAverageSpent']], on='CabinDeck', how='left')
    
    df["FamilyName"] = df["Name"].str.split(' ', expand=True)[1]
    no_relatives = df.groupby('FamilyName').size().reset_index(name='NoRelatives')
    df = df.merge(no_relatives, on='FamilyName', how='left')
    df["FamilySizeCat"] = pd.cut(df.NoRelatives, bins=[0, 2, 5, 10, 300], labels=['0-2', '3-5', '6-10', '11+'])
    
    return df

def objective_lgbm(trial, X_train, y_train):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
    num_leaves = trial.suggest_int('num_leaves', 2, 256)
    
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=42,
        force_row_wise=True
    )
    
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    return score

def objective_adaboost(trial, X_train, y_train):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0)
    
    model = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )
    
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    return score

def objective_calibrated_adaboost(trial, X_train, y_train):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0)
    method = trial.suggest_categorical('method', ['sigmoid', 'isotonic'])
    
    base_model = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )
    model = CalibratedClassifierCV(
        base_estimator=base_model,
        method=method,
        cv=3
    )
    
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    return score

def objective_calibrated_svc(trial, X_train, y_train):
    C = trial.suggest_float('C', 1e-6, 1e2, log=True)
    max_iter = trial.suggest_int('max_iter', 100, 10000)
    method = trial.suggest_categorical('method', ['sigmoid', 'isotonic'])
    
    base_model = LinearSVC(
        C=C,
        max_iter=max_iter,
        random_state=42
    )
    model = CalibratedClassifierCV(
        base_estimator=base_model,
        method=method,
        cv=3
    )
    
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    return score

def objective_calibrated_logistic(trial, X_train, y_train):
    C = trial.suggest_float('C', 1e-6, 1e2, log=True)
    solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
    max_iter = trial.suggest_int('max_iter', 100, 10000)
    method = trial.suggest_categorical('method', ['sigmoid', 'isotonic'])
    
    base_model = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=42
    )
    model = CalibratedClassifierCV(
        base_estimator=base_model,
        method=method,
        cv=3
    )
    
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    return score

def objective_stacking(trial, X_train, y_train, pipelines):
    C = trial.suggest_float('C', 1e-6, 1e2, log=True)
    solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
    max_iter = trial.suggest_int('max_iter', 100, 10000)
    
    final_estimator = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=42
    )
    
    stacking_ensemble = StackingClassifier(
        estimators=pipelines,
        final_estimator=final_estimator
    )
    
    score = cross_val_score(stacking_ensemble, X_train, y_train, cv=3, scoring='accuracy').mean()
    return score

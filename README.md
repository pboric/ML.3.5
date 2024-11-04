# Spaceship Titanic: Passenger Transport Prediction

## Project Overview

This project is centered around predicting whether passengers aboard the Spaceship Titanic were transported to an alternate dimension following an encounter with a spacetime anomaly. The prediction is a binary classification problem where the target variable is whether a passenger was transported or not.

## Dataset

The project utilizes three key datasets:

- **Train Dataset (`train.csv`)**: Contains records of passengers, including the target variable indicating whether they were transported.
- **Test Dataset (`test.csv`)**: Contains records of passengers without the target variable. This dataset is used to evaluate the model's performance on unseen data.
- **Sample Submission (`sample_submission.csv`)**: Provides the format for submitting predictions.

### Data Fields

- **PassengerId**: Unique ID for each passenger (format: `gggg_pp` where `gggg` is the group and `pp` is the passenger's number within the group).
- **HomePlanet**: The planet from which the passenger departed.
- **CryoSleep**: Indicates if the passenger was in suspended animation during the voyage.
- **Cabin**: The passenger's cabin number (format: `deck/num/side`).
- **Destination**: The destination planet.
- **Age**: The passenger's age.
- **VIP**: Whether the passenger paid for VIP services.
- **RoomService, FoodCourt, ShoppingMall, Spa, VRDeck**: Amounts billed at the Spaceship Titanic's amenities.
- **Name**: Passenger's first and last name.
- **Transported**: Target variable, indicating whether the passenger was transported to another dimension (True/False).

## Requirements

The following Python libraries are required:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- lazypredict
- optuna
- statsmodels
- scipy
- lightgbm

To install the required libraries, you can run:

```bash
pip install -r requirements.txt
```

## Project Structure

- **Data Loading**: Load the training and testing datasets.
- **Exploratory Data Analysis (EDA)**: Analyze and visualize the data to understand distributions and relationships.
- **Feature Engineering**: Create new features such as `CabinDeck`, `CabinSide`, `PassengerGroup`, and `IsAlone` from existing features.
- **Modeling**: Use various machine learning models to predict whether passengers were transported. Models include:
  - Random Forest
  - Decision Tree
  - LightGBM
  - Logistic Regression
- **Model Evaluation**: Evaluate models using metrics such as accuracy, F1-score, and ROC-AUC score.
- **Hyperparameter Tuning**: Use Optuna for hyperparameter optimization.

## How to Run

1. **Load Data**: Ensure that the `train.csv` and `test.csv` files are available in the working directory.
2. **Run the Script**: Execute the provided Python script. It will:
   - Load and preprocess the data.
   - Perform exploratory data analysis.
   - Train and evaluate various machine learning models.
   - Optimize the models using hyperparameter tuning.
3. **Generate Predictions**: The script will output predictions for the test dataset, which can be submitted for evaluation.

## Results

The results of the analysis include insights into which features are most important for predicting passenger transport and the performance metrics of different models. Passengers who were not alone, were in CryoSleep, or were on certain decks had higher probabilities of being transported.

## Author

Petar Krešimir Borić

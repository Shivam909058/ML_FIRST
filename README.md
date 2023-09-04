# README.md for Olympic Medal Prediction Project

## Overview

This project aims to predict the number of medals earned by a country in the Olympics. The predictions are based on past data and leverage a linear regression model for predictions.

## Data

The dataset `teams.csv` contains the following columns:

- `team`: Name of the team or country.
- `country`: Name of the country.
- `year`: Olympic year.
- `athletes`: Number of athletes from the country participating in that year.
- `age`: Average age of the athletes.
- `prev_medals`: Medals won by the country in the previous Olympic games.
- `medals`: Medals won by the country in the current Olympic year.

## Steps

1. **Data Cleaning**: 
    - Removed columns not related to the prediction.
    - Handled missing values by dropping rows with NaN values.

2. **Exploratory Data Analysis (EDA)**: 
    - Correlation matrix to identify relations between numeric columns.
    - Various plots were made using seaborn to visualize relations between variables.

3. **Data Splitting**:
    - Split data into train (data before 2012) and test (data from 2012 and onwards) datasets.

4. **Model Training**:
    - Used `LinearRegression` from scikit-learn.
    - Selected 'athletes' and 'prev_medals' as predictors.
    - Trained model using the training dataset.

5. **Prediction**:
    - Made predictions using the trained model on the test dataset.
    - Rounded off the predictions to the nearest integer since we can't have fractional medals.
    - Any negative prediction values were set to zero, as negative medals don't make sense.

6. **Error Analysis**:
    - Computed Mean Absolute Error (MAE) for predictions.
    - Compared MAE with the standard deviation of the medals to understand prediction accuracy.
    - Error by each team was computed and then error ratio was calculated for each team.

7. **Insights**:
    - Checked prediction accuracy for specific countries like USA, IND, DOM, ARM, and CAN.
    - Listed countries with an error ratio of less than 0.5, indicating higher prediction accuracy.

## Results

- The mean absolute error of the predictions is approximately 3.3 medals per team. 
- The error is less than the standard deviation of the medals in the dataset.
- Several countries like DOM, ARM, and CAN have been predicted with good accuracy.

## Libraries Used

- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `sklearn`

## How to Run

1. Ensure you have the required libraries installed.
2. Load the `teams.csv` data file.
3. Run the provided code sequentially to train the model and make predictions.

---

For any further details or queries, please reach out to the repository owner.

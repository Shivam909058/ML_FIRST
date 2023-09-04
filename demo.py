#%%
import pandas as pd
import numpy as np

#%%
teams = pd.read_csv('teams.csv')

#%%
teams  = teams[["team","country","year","athletes","age","prev_medals","medals"]]
teams

#%%
numeric_df = teams.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
correlation_matrix
#%%

print(teams.head())
print(teams.dtypes)

#%%

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Create the lmplot
sns.lmplot(x='athletes', y='medals', data=teams, fit_reg=True)

# Display the plot
plt.show()

#%%
sns.lmplot(x='age', y='medals', data=teams, fit_reg=True,ci=None)

#%%
sns.lmplot(x='prev_medals', y='medals', data=teams, fit_reg=True,ci=None)

#%%
sns.lmplot(x='year', y='medals', data=teams, fit_reg=True,ci=None)

#%%
teams.plot.hist(y='medals',bins=20)
#%%
# remove some outliers , mising values
teams[teams.isnull().any(axis=1)] # to find the rows with missing values

#%%
teams = teams.dropna() # drop the rows with missing values LIKE NAN

#%%
teams
#%%
train = teams[teams['year'] < 2012].copy() # here we are copying the data to train and test like it is the data of 2012 and before
test = teams[teams['year'] >= 2012].copy() # here we are copying the data to train and test like it is the data of 2012 and after

#%%
train.shape # to see the shape of the data
#%%
train.shape

#%%
from sklearn.linear_model import LinearRegression
reg = LinearRegression()



#%%
predictors = ['athletes','prev_medals'] # here we are taking the predictors as athletes and previous medals to predict the medals
target = 'medals'

#%%
reg.fit(train[predictors],train[target]) # here we are fitting the data to the model

#%%
predictions = reg.predict(test[predictors]) # here we are predicting the medals for the test data using the model we trained
#%%
test['predictions'] = predictions #rough prediction
#%%
test
#%%
test.loc[test['predictions'] < 0, 'predictions'] = 0 # removing negative values beacuse it is not possible to have negative medals
#%%
test['predictions'] = test['predictions'].round() # make it integer values and round it to nearest integer
test
#%%
from sklearn.metrics import mean_absolute_error
error =mean_absolute_error(test['medals'],test['predictions']) # we are finding the mean absolute error beacuse it is a regression problem and we are finding the error between the actual and predicted values of medals( actual - predicted) actual valuers=> test['medals'] and predicted values=> test['predictions']

#%%
error # here we are getting the error as 3.29876 it is the mean absolute error
#%%
# this means that on average we are off by 3.,,,, medals per team and the mean_absolute_error is 3.29876,,,,,,

#%%
teams.describe()["medals"] # here we are finding the mean of the medals column beacuse we are finding the error between the actual and predicted values of medals( actual - predicted) actual valuers=> test['medals'] and predicted values=> test['predictions']by finding the mean of the medals column we are finding the mean of the actual values of medals
#%%
# so the error is always than standard deviation of the data which is 33.6,,,,
test[test["team"]=="USA"]  # as we know that the error is always less than the standard deviation of the data , because the standard deviation of the data is 33.6 and the error is 3.29876 so the error is less than the standard deviation of the data , standard deviation of the data means that the data is spreaded over a large range so the error is less than the standard deviation of the data
#%%
test[test["team"]=="IND"]
#%%
error=(test["medals"]-test["predictions"]).abs() # here we are finding the error between the actual and predicted values of medals( actual - predicted) actual valuers=> test['medals'] and predicted values=> test['predictions'] and we are taking the absolute value of the error
#%%
error
#%%
error_by_team = error.groupby(test["team"]).mean() # now we are grouping the error by team and finding the mean of the error by team
#%%
error_by_team
#%%
medals_by_team=test["medals"].groupby(test["team"]).mean() # and here we are grouping the medals by team and finding the mean of the medals by team
#%%
error_ratio=error_by_team/medals_by_team # here we are finding the error ratio by dividing the error by team by the medals by team
#%%
error_ratio
#%%
#drop the teams with no medals or nan values
error_ratio[~pd.isnull(error_ratio)]

#%%
import numpy as np
error_ratio=error_ratio[np.isfinite(error_ratio)] # here we are removing the nan values from the error ratio column by using np.isfinite(error_ratio) and storing it in error_ratio

#%%
error_ratio
#%%
error_ratio.plot.hist() # here we are plotting the error ratio column it shows that the error ratio is between 0.0 to 0.5
#%%
# lets see an example with prediction of a country
#%%
test[test["team"]=="USA"]
#%%
# list of the countries with the error ratio less than 0.5 and have very accuarte prediactions
error_ratio[error_ratio<0.5]
#%%
teams=teams[teams["team"].isin(error_ratio[error_ratio<0.5].index)] # here we are taking the teams with the error ratio less than 0.5 and storing it in teams
#%%
test[test["team"]=="DOM"]
#%%
test[test["team"]=="ARM"]
#%%
test[test["team"]=="CAN"]
#%%

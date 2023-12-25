#-------------- Import libraries and packages --------------#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

#-------------- Import datasets --------------#
train_data = pd.read_csv("../input/nba-all-star-players-and-stats-1980-2022/final_data.csv")
test_data = pd.read_csv("../input/nba-all-star-players-and-stats-1980-2022/final_data.csv")
nba_all_stars = pd.read_csv("../input/nba-all-star-players-and-stats-1980-2022/final_data.csv")
#---------------------------------------------#

#-------------- Clean data / Feature engineering --------------#
year_map = {
    1980: 0,
    1981: 1,
    1982: 2,
    1983: 3,
    1984: 4,
    1985: 5,
    1986: 6,
    1987: 7,
    1988: 8,
    1989: 9,
    1990: 10,
    1991: 11,
    1992: 12,
    1993: 13,
    1994: 14,
    1995: 15,
    1996: 16,
    1997: 17,
    1998: 18,
    1999: 19,
    2000: 20,
    2001: 21,
    2002: 22,
    2003: 23,
    2004: 24,
    2005: 25,
    2006: 26,
    2007: 27,
    2008: 28,
    2009: 29,
    2010: 30,
    2011: 31,
    2012: 32,
    2013: 33,
    2014: 34,
    2015: 35,
    2016: 36,
    2017: 37,
    2018: 38,
    2019: 39,
    2020: 40,
    2021: 41,
    2022: 42
}

train_data['year'] = train_data['year'].map(year_map)
test_data['year'] = test_data['year'].map(year_map)

train_data.loc[(train_data['X'] == -120.5) | (train_data['Y'] == 90.0), ['X', 'Y']] = None
test_data.loc[(test_data['X'] == -120.5) | (test_data['Y'] == 90.0), ['X', 'Y']] = None

imputer = SimpleImputer(strategy='mean')
# Feature engineering for x and y values rotated 30, 45, 60 degrees on train_data
train_data[['X', 'Y']] = imputer.fit_transform(train_data[['X', 'Y']])
test_data[['X', 'Y']] = imputer.fit_transform(test_data[['X', 'Y']])

train_data['X_rotated_30'] = train_data['X'] * 0.866 - train_data['Y'] * 0.5
train_data['Y_rotated_30'] = -train_data['X'] * 0.5 + train_data['Y'] * 0.866

train_data['X_rotated_45'] = train_data['X'] * 0.707 + train_data['Y'] * 0.707
train_data['Y_rotated_45'] = train_data['X'] * 0.707 + train_data['Y'] * 0.707

train_data['X_rotated_60'] = train_data['X'] * 0.5 + train_data['Y'] * 0.866
train_data['Y_rotated_60'] = -train_data['X'] * 0.866 + train_data['Y'] * 0.5

train_data['R'] = np.sqrt(train_data['X']**2 + train_data['Y']**2)
train_data['Theta'] = np.arctan2(train_data['Y'], train_data['X'])

# Feature engineering for x and y values rotated 30, 45, 60 degrees on test_data
test_data['X_rotated_30'] = test_data['X'] * 0.866 - test_data['Y'] * 0.5
test_data['Y_rotated_30'] = -test_data['X'] * 0.5 + test_data['Y'] * 0.866

test_data['X_rotated_45'] = test_data['X'] * 0.707 + test_data['Y'] * 0.707
test_data['Y_rotated_45'] = test_data['X'] * 0.707 + test_data['Y'] * 0.707

test_data['X_rotated_60'] = test_data['X'] * 0.5 + test_data['Y'] * 0.866
test_data['Y_rotated_60'] = -test_data['X'] * 0.866 + test_data['Y'] * 0.5

test_data['R'] = np.sqrt(test_data['X']**2 + test_data['Y']**2)
test_data['Theta'] = np.arctan2(test_data['Y'], test_data['X'])
#--------------------------------------------------------------#

#-------------- Split data into features / Target / Train / Validation --------------#
X = train_data[['Y', 'X', 'DayOfWeek', 'PdDistrict', 'X_rotated_30', 'Y_rotated_30', 'X_rotated_45', 'Y_rotated_45', 'X_rotated_60', 'Y_rotated_60', 'R', 'Theta']]
y = train_data['fg3a']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=['X'])
#------------------------------------------------------------------------------------#

#-------------- More data cleaning --------------#
numeric_features = ['Y', 'X', 'DayOfWeek', 'X_rotated_30', 'Y_rotated_30', 'X_rotated_45', 'Y_rotated_45', 'X_rotated_60', 'Y_rotated_60', 'R', 'Theta']
categorical_features = ['pts'] # Change to points or years?

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
#------------------------------------------------#

#-------------- See how the number of three pointers changed over time --------------#
def three_point_evolution():
    
    # Assuming 'year' and 'fg3a' are column in your DataFrame
    X = nba_all_stars[['year']]
    y = nba_all_stars[['fg3a']]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Lower random state?
    
    # Use scikit-learn for some machine learning task (ie. regression)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Plot the actus vs predicted values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_test['year'], y=y_test, color='blue', label='Actual')
    sns.scatterplot(x=X_test['year'], y=predictions, color='red', label='Predicted')
    plt.xlabel('Year')
    plt.ylabel(('3-Pointers Attempted'))
    plt.title('Actual vs Predicted 3-Pointers Attempted by NBA All Stars')
    plt.legend()
    plt.show()
    return
#--------------------------------------------------------------------#

#-------------- Trains a linear regression model --------------#
def run_linear_regression():
    # FIX!!!
    lr_model = Pipeline(steps = [('preprocessor', preprocessor), ('classifier', LogisticRegression(random_state = 42, max_iter = 1000))]) # Lower random_state to improve Logistic Regression?
    
    lr_model.fit(X_train, y_train)
    lr_val_pred = lr_model.predict_proba(X_val)
    
    lr_model.fit(X, y)
    lr_pred = lr_model.predict_proba(test_data[['Y', 'X', 'DayOfWeek', 'X_rotated_30', 'Y_rotated_30', 'X_rotated_45', 'Y_rotated_45', 'X_rotated_60', 'Y_rotated_60', 'R', 'Theta']])
    
    class_names = lr_model.named_steps['classifier'].classes_
    lr_df = pd.DataFrame(lr_pred, columns = class_names)
    lr_df.insert(0, 'Incident_ID', test_data)
    
    lr_df.to_csv('logistic_regression.csv', index = False)
    return
#--------------------------------------------------------------------#



# Call models
# run_linear_regression()
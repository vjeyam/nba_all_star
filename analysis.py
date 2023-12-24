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

#-------------- Import datasets --------------#
nba_all_stars = pd.read_csv("../input/nba-all-star-players-and-stats-1980-2022/final_data.csv")

#-------------- Clean data / Feature engineering --------------#

#-------------- Split data into features / Target / Train / Validation --------------#

#-------------- See how the number of three pointers changed over time --------------#
def three_point_evolution():
    
    # Assuming 'year' and 'fg3a' are column in your DataFrame
    x = nba_all_stars[['year']]
    y = nba_all_stars[['fg3a']]
    
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Use scikit-learn for some machine learning task (ie. regression)
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    
    # Make predictions
    predictions = model.predict(x_test)
    
    # Plot the actus vs predicted values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_test['year'], y=y_test, color='blue', label='Actual')
    sns.scatterplot(x=x_test['year'], y=predictions, color='red', label='Predicted')
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
    lr_model = Pipeline(steps = [('preprocessor', preprocessor),
                                 ('classifier', LogisticRegression(random_state = 42, max_iter = 1000))])
    
    lr_model.fit(x_train, y_train)
    lr_val_pred = lr_model.predict_proba(x_val)
    
    lr_model.fit(x, y)
    lr_pred = lr_model.predict_proba(test_data[['Y', 'X', 'DayOfWeek', 'X_rotated_30', 'Y_rotated_30', 'X_rotated_45', 'Y_rotated_45', 'X_rotated_60', 'Y_rotated_60', 'R', 'Theta']])
    
    class_names = lr_model.named_steps['classifier'].classes_
    lr_df = pd.DataFrame(lr_pred, columns = class_names)
    lr_df.insert(0, 'Incident_ID', test_data_index)
    
    lr_df.to_csv('logistic_regression.csv', index = False)
    return
   
#--------------------------------------------------------------------#

# Call models
# three_point_evolution()
#-------------- Import libraries and packages --------------#
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

#-------------- Import datasets --------------#
nba_all_stars = pd.read_csv("../input/nba-all-star-players-and-stats-1980-2022/final_data.csv")

#-------------- Clean data / Feature engineering --------------#

#-------------- Split data into features / Target / Train / Validation --------------#

#-------------- Trains a random forest classifier model --------------#
def three_point_evolution() -> None:
    
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
#--------------------------------------------------------------------#

# Call models
three_point_evolution()
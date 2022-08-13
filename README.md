# Prediction-using-supervised-ML
Predict percentage of student using number of hours studied
#import packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

# import data set
my_df = pd.read_csv("video_game_data.csv")


#split data into input and output object
x = my_df.drop(["completion_time"], axis= 1)
y = my_df["completion_time"]


#split data into training set and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 42)

#instantiate the model object
regressor = RandomForestRegressor(random_state=42)


#train the model

regressor.fit(x_train, y_train)
#acess model accuracy

y_pred = regressor.predict(x_test)

prediction_comparison = pd.DataFrame({"actual" : y_test, "prediction" : y_pred})
#how acturatly model worked
r2_score(y_test, y_pred)
print(r2_score(y_test, y_pred))

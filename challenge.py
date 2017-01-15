import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#guess index
guess = 63

#import data
dataframe = pd.read_csv('challenge_dataset.txt', names=['x_values','y_values'])
x_values = dataframe[['x_values']]
y_values = dataframe[['y_values']]

#create linear regression
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#extract data from da
x_column = dataframe['x_values']
y_column = dataframe['y_values']

#Using guess as a random index
print("X-Value at guess index: " + str(x_column[guess]))
print("Actual Y-Value from dataset: " + str(y_column[guess]))
predicted = body_reg.predict(x_values)
print("Predicted Y-Value from linear regression: " + str(*predicted[guess]))
difference = y_column[guess] - predicted[guess]
print("Difference between Y-Values with dataset value as reference: " + str(*difference))

#create graph to visualize 
plt.scatter(x_values, y_values)
plt.plot(x_values, predicted)
plt.show()
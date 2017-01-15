import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


def find_error(x_values, y_values, reg_line):
    predictions = [float(reg_line.predict(x.reshape(-1,1))) for x in x_values.values]
    differences = [float(y) - predict for y,predict in zip(y_values.values, predictions)]
    sqrs = [diff**2 for diff in differences]
    sum_sqrs = sum(sqrs)
    n = x_values.size

    return (sum_sqrs/n)**.5


data = pd.read_csv('challenge_data.txt', dtype={'x':float, 'y':float})
x_values = data[['x']]
y_values = data[['y']]


line = linear_model.LinearRegression()
line.fit(x_values, y_values)
standard_error = find_error(y_values, x_values, line)

#Prints the estimation line and standard error
print("y' = {0}x + {1}".format(line.coef_[0][0], line.intercept_[0]))
print("Standard Error = {0}".format(standard_error))

plt.scatter(x_values, y_values)
plt.plot(x_values, line.predict(x_values))
plt.show()

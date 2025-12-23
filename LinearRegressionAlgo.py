import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
lr = linear_model.LinearRegression()

data = pd.read_csv(r"D:\Internship\train.csv")
X = data["x"].values
X = X.reshape(X.size, 1)
Y = data["y"].values
Y = Y.reshape(Y.size, 1)

lr.fit(X, Y)

data1 = pd.read_csv(r"D:\Internship\test.csv")
X_test = data1["X"].values
X_test = X_test.reshape(X_test.size, 1)
Y_test = data1["Y"].values
Y_test = Y_test.reshape(Y_test.size, 1)

prediction = lr.predict(X_test)
print("Mean Squared Error: %.2f" % mean_squared_error(Y_test, prediction))
print("Coefficients of determination: %.2f" % r2_score(Y_test, prediction))

colors = np.random.randint(10, 100, size = (1, len(Y)))
plt.scatter(X, Y, c = colors, cmap = "Pastel1", label = "Y")
plt.plot(X_test, prediction, color = "lavender", label = "Predictions")
plt.show()

i = 0

print(lr.score(X,Y))
print(lr.score(X_test, Y_test))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data  = pd.read_csv(r"C:\Users\soban\Downloads\swedish_insurance.csv")
X = data["X"].values
Y = data["Y"].values
X = np.c_[np.ones(len(X)), X]
X = X.T

def hypothesis (thetha, X):
    predictions = np.matmul(thetha.T,X)
    return predictions

def compute_cost (predictions, actual):
    m = len(actual)
    errors = predictions - actual
    squared = errors*errors
    mse = np.sum(squared)/(2*m)
    return mse

def gradient_descent(X, Y, learning_rate, iterations):
    actual = Y
    costs = []
    m = len(actual)
    thetha = np.zeros(X.shape[0])
    for i in range(iterations):
        predictions = hypothesis(thetha, X)
        errors = predictions - actual
        costs.append(compute_cost(predictions, actual))
        gradients = (1/m)*np.matmul(X,errors)
        thetha = thetha - (learning_rate*gradients)
    return thetha, costs

learning_rate = 0.0005
iterations = 1000
thetha,costs = gradient_descent(X, Y, learning_rate, iterations)

plt.plot(np.arange(1000), costs, label = 'Costs')

predictions = np.matmul(thetha.T, X)
colors = np.random.randint(10, 100, size = (1, len(Y)))
plt.scatter(X[1,:], Y, c = colors, cmap = "Pastel1", label = 'Actuals')

plt.plot(X[1,:], predictions, color = 'lavender', label = 'Predictions')
plt.axis([0, 120, 0, 110])

plt.xlabel("Training Data")
plt.ylabel("Costs")
plt.title("Linear Regression for Single Variable")

##num = int(input("Please enter number of ages: "))
##lis = []
##for i in range(num):
##    ins = int(input("Enter age: "))
##    lis.append(ins)
##
##lis = np.array(lis)
##new_insurance = np.vstack((np.ones([1, lis.size]), lis))
##y_real = hypothesis(thetha, new_insurance)
##
##j = 0
##for insurance in lis:
##    print("The insurance for age: ", insurance, " is ", y_real[j])
##    j  += 1
##

data1  = pd.read_csv("C:\\Users\\soban\\Downloads\\test.csv")
X1 = data["X"].values
Y_test = data["Y"].values
X_test = np.c_[np.ones(len(X1)), X1]
X_test = X_test.T

colors = np.random.randint(10, 100, size = (1, len(Y)))
plt.scatter(X_test[1,:], Y_test, c = colors, cmap = "gist_rainbow", label="Testing values")
prediction_t = hypothesis(thetha, X_test)
plt.legend()
plt.show()

i = 0

for pred in X1:
    print("The predicted values for ", X1[i], " are : ", prediction_t[i])
    i = i + 1
    print()

N = Y_test.size
MAE = (1/N)*np.sum(np.abs(prediction_t - Y_test))
print(MAE)

MSE = (1/N)*np.sum(np.square(prediction_t - Y_test))
print(MSE)

RMSE = np.sqrt(MSE)
print(RMSE)

R2 = 1 - ((np.sum(np.square(prediction_t - Y_test)))/(np.sum(np.square(prediction_t - np.mean(Y_test)))))
print(R2)

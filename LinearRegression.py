from csv import reader
import sys
import numpy as np, math
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

#sns 

#12 columns in the wine dataset, where the last one is the target (Quality)
#Volatile Acidity = Column 2, Alcohol Content (Alcohol) = Column 11 

#Load the CSV file using pandas library
dataset = pd.read_csv('wine.csv')

#Define the X and Y variables for our hypothesis
X1 = dataset['volatileacidity']
X2 = dataset['alcohol']
Y = dataset['quality']

#Split the dataset by class values, returns a dictionary. from exercise
def seperateByClass(dataset):
    seperated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in seperated):
            seperated[class_value] = list()
        seperated[class_value].append(vector)
    return seperated

model = LinearRegression()
model.fit(dataset[['volatileacidity', 'alcohol']], dataset['quality'])


#retrieve various information about the dataset for our use case..
#we want to find r^2, MSE, SSE, AIC, BIC values for our linear regression model
#AIC and BIC calculations can be found through the exercises we did in class
#function to calculate r^2 which is 1 - Sum(yActual - yPredicted)^2 / Sum(yActual - yMean)^2

def rSquared(yActual, yPredicted):
    yMean = np.mean(yActual)
    sumSquaredNumerator = np.sum((yActual - yMean) ** 2)
    sumSquaredDenominator = np.sum((yActual - yPredicted) ** 2)
    return 1 - (sumSquaredDenominator / sumSquaredNumerator)


#function to calculate the Mean Squared Error (MSE)
def meanSquaredError(yActual, yPredicted):
    return np.mean((yActual - yPredicted) ** 2)


#function to calculate the Sum Squared Error (SSE)
def sumSquaredError(yActual, yPredicted):
    return np.sum((yActual - yPredicted) ** 2)

#function to calculate the AIC values
def calculateAIC(y, X):
    n = y.shape[0]
    if X is None or X.size == 0:
        mu = y.mean()
        resid = y - mu
        s2 = max(float(np.dot(resid, resid) / n), 1e-12)
        ll = -0.5 * n * (math.log(2*math.pi*s2) + 1)
        k = 2
        return ll - k
    X_ = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
    resid = y - X_.dot(beta)
    s2 = max(float(np.dot(resid, resid) / n), 1e-12)
    ll = -0.5 * n * (math.log(2*math.pi*s2) + 1)
    k = X_.shape[1] + 1
    return ll - k

#function to calculate the BIC values
def calculateBIC(y, X):
    n = y.shape[0]
    if X is None or X.size == 0:
        mu = y.mean()
        resid = y - mu
        s2 = max(float(np.dot(resid, resid) / n), 1e-12)
        ll = -0.5 * n * (math.log(2*math.pi*s2) + 1)
        k = 2
        return ll - 0.5 * k * math.log(n)
    X_ = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
    resid = y - X_.dot(beta)
    s2 = max(float(np.dot(resid, resid) / n), 1e-12)
    ll = -0.5 * n * (math.log(2*math.pi*s2) + 1)
    k = X_.shape[1] + 1
    return ll - 0.5 * k * math.log(n)


fig = plt.figure(figsize=(6,5))
axes = fig.add_subplot(111, projection='3d')


def main():
    print("Loading dataset...")
    print(f"Dataset loaded. Number of rows: ", len(dataset) + 1)
    #plt.scatter(dataset['volatileacidity'], dataset['alcohol'], c=dataset['quality'], cmap='viridis')
    axes.scatter(X1, X2, Y, c=Y, cmap='viridis')
    axes.set_xlabel('Volatile Acidity')
    axes.set_ylabel('Alcohol Content')
    axes.set_zlabel('Quality')
    plt.title('3D Scatter Plot of Wine Quality')
    plt.show()
    print(f"R^2: {rSquared(Y, model.predict(dataset[['volatileacidity', 'alcohol']]))}")
    print(f"MSE: {meanSquaredError(Y, model.predict(dataset[['volatileacidity', 'alcohol']]))}")
    print(f"SSE: {sumSquaredError(Y, model.predict(dataset[['volatileacidity', 'alcohol']]))}")
    print(f"AIC: {calculateAIC(Y, dataset[['volatileacidity', 'alcohol']])}")
    print(f"BIC: {calculateBIC(Y, dataset[['volatileacidity', 'alcohol']])}")

if __name__ == "__main__":
    main()


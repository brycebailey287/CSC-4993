from csv import reader
import sys
import numpy as np, math
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


#12 columns in the wine dataset, where the last one is the target (Quality)
#Volatile Acidity = Column 2, Alcohol Content (Alcohol) = Column 11 


#Load the CSV file using pandas library
dataset = pd.read_csv('wine.csv')
print(f"Dataset Shape: {dataset.shape}")
#Define the X and Y variables for our hypothesis
X1 = dataset['volatileacidity']
print(f"Shape of X: {X1.shape}")
X2 = dataset['alcohol']
Y = dataset['quality']
print(f"Shape of Y: {Y.shape}")

X = np.column_stack((X1, X2))
#X = np.column_stack(X1)

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

#xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Shape of X: {X.shape}")  # Should be (n_samples, n_features)
print(f"Shape of Y: {Y.shape}")  # Should be (n_samples,)

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(xTrain, yTrain)

yPredicted = model.predict(xTest)
yActual = yTest

mse = mean_squared_error(yActual, yPredicted)
print(f"MSE from sklearn: {mse}")

r2Score = r2_score(yActual, yPredicted)
print(f"R^2 from sklearn: {r2Score}")

sseFromYActual = mse * len(yActual)
print(f"SSE from sklearn: {sseFromYActual}")

print("yActual: ", yActual)
print("yPredicted: ", yPredicted)
#retrieve various information about the dataset for our use case..
#we want to find r^2, MSE, SSE, AIC, BIC values for our linear regression model
#AIC and BIC calculations can be found through the exercises we did in class

#function to calculate r^2 which is 1 - Sum(yActual - yPredicted)^2 / Sum(yActual - yMean)^2 -> 1 - SSE / SST
def rSquared(yActual, yPredicted):
    yMean = np.mean(yActual)
    sumSquaredNumerator = np.sum((yActual - yMean) ** 2)
    sumSquaredDenominator = np.sum((yActual - yPredicted) ** 2)
    return 1 - (sumSquaredDenominator / sumSquaredNumerator)


#function to calculate the Mean Squared Error (MSE)
def meanSquaredError(yActual, yPredicted):
    return np.mean((yActual - yPredicted) ** 2)
    #return mean_squared_error(yActual, yPredicted)


#function to calculate the Sum Squared Error (SSE)
def sumSquaredError(yActual, yPredicted):
    return np.sum((yActual - yPredicted) ** 2)


#function to calculate the Sum Squared Total (SST)
def sumSquaredTotal(yActual):
    yMean = np.mean(yActual)
    return np.sum((yActual - yMean) ** 2)

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


def main():
    print("Loading dataset...")
    print(f"Dataset loaded. Number of rows: ", len(dataset) + 1)
    print(f"R^2: {rSquared(Y, model.predict(X))}")
    print(f"MSE: {meanSquaredError(Y, model.predict(X))}")
    print(f"SSE: {sumSquaredError(Y, model.predict(X))}")
    print(f"AIC: {calculateAIC(Y, X)}")
    print(f"BIC: {calculateBIC(Y, X)}")

    # Create a meshgrid for the plane
    x1_range = np.linspace(X1.min(), X1.max(), 20)
    x2_range = np.linspace(X2.min(), X2.max(), 20)
    #x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

    # Predict Y values for the meshgrid
    #mesh_points = np.column_stack((x1_mesh.ravel(), x2_mesh.ravel()))
    mesh_points = np.column_stack((x1_mesh.ravel(), x2_mesh.ravel()))
    y_mesh = model.predict(mesh_points).reshape(x1_mesh.shape)

    # Plot the scatter plot and the regression plane
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(X1, X2, Y, c=Y, cmap='viridis', alpha=0.6, label='Data Points')
    #ax.scatter(X1, Y, c=Y, cmap='viridis', alpha=0.6, label='Data Points')

    # Regression plane
    ax.plot_surface(x1_mesh, x2_mesh, y_mesh, alpha=0.3, color='blue', label='Regression Plane')
    #ax.plot_surface(x1_mesh, y_mesh, alpha=0.3, color='red', label='Regression Plane')
    # Labels and title  
    ax.set_xlabel('X1 (Volatile Acidity)')
    ax.set_ylabel('X2 (Alcohol Content)')
    ax.set_zlabel('Y (Quality)')
    ax.set_title('3D Scatter Plot with Regression Plane')

    plt.show()

if __name__ == "__main__":
    main()


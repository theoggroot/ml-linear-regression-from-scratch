# Linear Regression From Scratch
# -------------------------------------------
# This project implements linear regression
# without using any machine learning libraries.
#
# Concepts implemented:
# - prediction using y = wx + b
# - mean squared error cost function
# - gradient descent optimization
# -------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# STEP 1: GENERATE DATASET
# -------------------------------

# Fix random seed so results are reproducible
np.random.seed(0)

# Create input values (1 → 49)
X = np.arange(1,50)

# Generate random noise
# Real world data always has randomness
noise = np.random.normal(0,5,len(X))

# True relationship is y = 2x
# But we add noise to simulate real data
Y = 2*X + noise


# -------------------------------
# STEP 2: INITIALIZE PARAMETERS
# -------------------------------

# Model parameters
# w = slope
# b = intercept

w = 0
b = 0


# -------------------------------
# STEP 3: PREDICTION FUNCTION
# -------------------------------

def predict(X, w, b):

    """
    Predict output using linear model

    y = wx + b
    """

    return w*X + b


# -------------------------------
# STEP 4: COST FUNCTION
# -------------------------------

def compute_cost(Y, Y_pred):

    """
    Calculate Mean Squared Error

    J = (1/n) * Σ (y_pred - y_actual)^2
    """

    n = len(Y)

    cost = (1/n) * np.sum((Y_pred - Y)**2)

    return cost


# -------------------------------
# STEP 5: GRADIENT DESCENT
# -------------------------------

def gradient_descent(X, Y, w, b, learning_rate):

    """
    Update model parameters to reduce error
    """

    n = len(X)

    # predict values
    Y_pred = predict(X, w, b)

    # calculate gradients
    dw = (-2/n) * np.sum(X * (Y - Y_pred))
    db = (-2/n) * np.sum(Y - Y_pred)

    # update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

    return w, b


# -------------------------------
# STEP 6: TRAINING LOOP
# -------------------------------

learning_rate = 0.0005
epochs = 1000

for i in range(epochs):

    # update parameters
    w, b = gradient_descent(X, Y, w, b, learning_rate)

    # print cost every 100 iterations
    if i % 100 == 0:

        Y_pred = predict(X, w, b)

        cost = compute_cost(Y, Y_pred)

        print(f"Epoch {i} | Cost: {cost:.2f}")


# -------------------------------
# STEP 7: FINAL MODEL
# -------------------------------

Y_pred = predict(X, w, b)

print("\nFinal Parameters")
print("Weight (w):", w)
print("Bias (b):", b)


# -------------------------------
# STEP 8: VISUALIZATION
# -------------------------------

plt.scatter(X, Y, label="Actual Data")

plt.plot(X, Y_pred, color="red", label="Regression Line")

plt.xlabel("X")
plt.ylabel("Y")

plt.title("Linear Regression From Scratch")

plt.legend()

plt.show()
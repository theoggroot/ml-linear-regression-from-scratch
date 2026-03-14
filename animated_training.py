# Linear Regression Training Animation
# ------------------------------------
# This script visualizes how gradient descent
# learns the best regression line step by step.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ---------------------------
# STEP 1: GENERATE DATASET
# ---------------------------

np.random.seed(0)

X = np.arange(1,50)

noise = np.random.normal(0,5,len(X))

Y = 2*X + noise


# ---------------------------
# STEP 2: INITIAL PARAMETERS
# ---------------------------

# model starts with a bad guess
w = 0
b = 0


# ---------------------------
# STEP 3: PREDICTION FUNCTION
# ---------------------------

def predict(X, w, b):

    """
    Linear regression model

    y = wx + b
    """

    return w*X + b


# ---------------------------
# STEP 4: COST FUNCTION
# ---------------------------

def compute_cost(Y, Y_pred):

    """
    Mean Squared Error
    """

    n = len(Y)

    cost = (1/n) * np.sum((Y_pred - Y)**2)

    return cost


# ---------------------------
# STEP 5: GRADIENT DESCENT
# ---------------------------

def gradient_descent(X, Y, w, b, lr):

    """
    Update parameters using gradient descent
    """

    n = len(X)

    Y_pred = predict(X, w, b)

    # derivative wrt w
    dw = (-2/n) * np.sum(X*(Y - Y_pred))

    # derivative wrt b
    db = (-2/n) * np.sum(Y - Y_pred)

    # update parameters
    w = w - lr * dw
    b = b - lr * db

    return w, b


# ---------------------------
# STEP 6: TRAINING SETTINGS
# ---------------------------

learning_rate = 0.0005
epochs = 200


# ---------------------------
# STEP 7: CREATE GRAPH
# ---------------------------

fig, ax = plt.subplots()

# scatter plot of data
ax.scatter(X,Y)

line, = ax.plot(X, predict(X,w,b), color="red")

ax.set_title("Gradient Descent Learning")
ax.set_xlabel("X")
ax.set_ylabel("Y")


# ---------------------------
# STEP 8: ANIMATION FUNCTION
# ---------------------------

def update(frame):

    global w,b

    # update model parameters
    w,b = gradient_descent(X,Y,w,b,learning_rate)

    # compute new prediction
    y_pred = predict(X,w,b)

    # update regression line
    line.set_ydata(y_pred)

    return line,


# ---------------------------
# STEP 9: RUN ANIMATION
# ---------------------------

ani = FuncAnimation(fig, update, frames=epochs, interval=50)

plt.show()
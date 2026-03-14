# Linear Regression From Scratch

This project implements **Linear Regression from scratch using Python**, without using machine learning libraries such as scikit-learn.

The goal of this project is to understand the **mathematics and optimization process behind machine learning algorithms** by implementing every step manually.

---

# Project Overview

Linear Regression is one of the simplest and most fundamental algorithms in machine learning. It models the relationship between an input variable and an output variable by fitting a straight line to the data.

The model assumes a linear relationship of the form:

y = wx + b

Where:

- **x** → input feature  
- **y** → predicted output  
- **w** → weight (slope of the line)  
- **b** → bias (intercept)

The objective is to learn the best values of **w** and **b** that minimize prediction error.

---

# Dataset

Instead of using a real dataset, we generate a **synthetic dataset with noise** to simulate real-world data.

True relationship:

y = 2x

But the dataset contains randomness:

y ≈ 2x + noise

This makes the learning process more realistic.

---

# Algorithm

The learning process follows these steps:

1. Initialize model parameters  
   w = 0  
   b = 0  

2. Make predictions using

   y_pred = wx + b

3. Compute prediction error using **Mean Squared Error**

   J = (1/n) Σ (y_pred − y_actual)²

4. Update parameters using **Gradient Descent**

   w = w − α * (∂J/∂w)  
   b = b − α * (∂J/∂b)

5. Repeat until the error is minimized.

---

# Concepts Implemented

This project implements the following machine learning fundamentals:

- Linear regression model
- Prediction function
- Mean Squared Error (MSE)
- Gradient descent optimization
- Training loop
- Synthetic dataset generation
- Data visualization
- Animated training visualization

---

# Visualization

The project visualizes:

• The dataset  
• The learned regression line  
• An animation showing how gradient descent gradually finds the best fit line

This helps understand how machine learning models **learn iteratively**.

---

# Project Structure

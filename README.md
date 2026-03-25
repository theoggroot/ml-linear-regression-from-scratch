# Linear Regression From Scratch

This project implements **Linear Regression from scratch using Python**, without using any machine learning libraries such as scikit-learn.

The goal of this project is to understand how machine learning works internally — including prediction, error calculation, and optimization using gradient descent.

---

# 🧠 What is Linear Regression?

Linear Regression is a machine learning algorithm used to model the relationship between an input variable (X) and an output variable (Y).

It assumes a linear relationship:

y = wx + b

Where:

- x → input feature  
- y → predicted output  
- w → weight (slope of the line)  
- b → bias (intercept)  

The objective is to find the best values of **w and b** that minimize prediction error.

---

# 📊 Dataset

This project uses a **synthetic dataset with noise** to simulate real-world data.

True relationship:

y = 2x

Actual data:

y ≈ 2x + noise

This allows the model to learn patterns from imperfect data.

---

# ⚙️ Step-by-Step Working

## 1. Initialize Parameters

Start with initial values:

w = 0  
b = 0  

This is just a random guess.

---

## 2. Prediction

Use the model:

y_pred = wx + b

The model predicts output values using current parameters.

---

## 3. Compute Error (Cost Function)

We measure how wrong the model is using **Mean Squared Error (MSE)**:

J = (1/n) Σ (y_pred − y_actual)²

Why squared error?

- Prevents negative values canceling out  
- Penalizes large errors more  

---

## 4. Gradient Descent (Learning)

Update parameters using:

w = w − α * (∂J/∂w)  
b = b − α * (∂J/∂b)

Where:

- α = learning rate (step size)  
- Derivatives indicate direction to reduce error  

---

## 5. Training Loop

Repeat:

predict → calculate error → update parameters

for multiple iterations (epochs) until the model improves.

---

## 6. Final Model

After training, the model learns:

w ≈ 2  
b ≈ 0  

Which approximates:

y ≈ 2x

---

# 📈 Visualization

The project includes:

- Scatter plot of dataset  
- Regression line  
- Animated visualization showing how the model learns step-by-step  

---

# 🧪 Libraries Used

## NumPy

Used for numerical computations.

- np.arange() → generate input data  
- np.random.normal() → add noise  
- np.sum() → compute cost and gradients  
- vector operations → efficient calculations  

---

## Matplotlib

Used for visualization.

- plt.scatter() → plot data points  
- plt.plot() → draw regression line  
- plt.show() → display graph  

---

## Matplotlib Animation

Used for animation.

- FuncAnimation → shows how the regression line updates during training  

---

# 📁 Project Structure

ml-linear-regression-from-scratch
│
├── data
│   └── sample_data.csv
│
├── notebooks
│
├── src
│
├── main.py
├── animated_training.py
├── requirements.txt
├── README.md

---

# ▶️ How to Run

Clone the repository:

git clone https://github.com/your-username/ml-linear-regression-from-scratch.git

Go to the project folder:

cd ml-linear-regression-from-scratch

Install dependencies:

pip install -r requirements.txt

Run the main model:

python main.py

Run animation:

python animated_training.py

---

# 🎯 What You Learn

- Linear regression fundamentals  
- Cost function (Mean Squared Error)  
- Gradient descent optimization  
- Model training process  
- Data visualization  

---

# 🚀 Future Improvements

- Multiple Linear Regression  
- Feature scaling  
- Polynomial Regression  
- Logistic Regression from scratch  
- Neural Networks from scratch  

---

# ✨ Conclusion

This project shows that machine learning is built on:

mathematics + optimization + iteration

Understanding these fundamentals is essential for advanced AI and deep learning.

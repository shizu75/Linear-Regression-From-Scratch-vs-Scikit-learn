# Linear Regression: From Scratch vs Scikit-learn

## Overview
This repository presents a complete implementation of single-variable Linear Regression using two approaches: a from-scratch implementation based on gradient descent and matrix operations, and a standard implementation using Scikit-learn’s LinearRegression model. The project emphasizes conceptual understanding, optimization, evaluation, and visualization.

## Datasets
The project uses separate CSV files for training and testing. Each dataset contains one input feature (X/x) and one target variable (Y/y). File paths should be updated locally before execution.

## From-Scratch Implementation
The first part implements Linear Regression without using machine learning libraries. A bias term is added manually to the feature matrix. The hypothesis function computes predictions using matrix multiplication. The cost function is defined as Mean Squared Error (MSE), and gradient descent is implemented to iteratively update parameters. Training runs for a fixed number of iterations with a specified learning rate, while tracking cost values to visualize convergence.

## Training and Visualization
During training, a cost-versus-iterations plot is generated to show optimization behavior. A scatter plot of actual data points is displayed alongside the regression line produced by the trained model.

## Testing and Evaluation (From Scratch)
The trained model is evaluated on a separate test dataset. Performance metrics are computed manually, including Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²). Predicted values are printed for each test input.

## Scikit-learn Implementation
The second part uses Scikit-learn’s LinearRegression model. Training and testing data are reshaped appropriately and passed to the model. Predictions are generated on the test set, and evaluation metrics including Mean Squared Error and R² score are computed using built-in functions. Model scores on training and testing data are also reported.

## Comparison and Visualization
Scatter plots and regression lines are used to visually compare predictions. This enables a clear comparison between the custom gradient descent approach and the optimized library-based implementation.

## Technologies Used
Python, Pandas, NumPy, Matplotlib, Scikit-learn

## How to Run
Clone the repository, install required dependencies, update dataset paths in the scripts, and run the Python files.

## Applications
Predictive modeling, trend analysis, insurance cost estimation, biomedical data modeling, regression analysis, and machine learning education.

## Future Work
Extensions may include multivariate linear regression, hyperparameter tuning, regularization methods, alternative optimization strategies, and application to larger real-world datasets.

## Author
Soban Saeed
GitHub: https://github.com/shizu75

## License
MIT

import numpy as np

data = np.genfromtxt('data.csv', delimiter=',')
X = data[:, :5]
y = data[:, 5]

start_term = np.ones((X.shape[0], 1));
X_poly = np.column_stack((start_term, X))
for i in range(5):
    for j in range(i + 1, 5):
        X_poly = np.column_stack((X_poly, X[:, i] * X[:, j]))
X_poly = np.column_stack((X_poly, X**2))

X_mean = np.mean(X_poly[:,1:], axis=0)
X_std = np.std(X_poly[:,1:], axis=0)
X_scaled = (X_poly[:,1:] - X_mean) / X_std
X_scaled = np.column_stack((start_term, X_scaled))
y_mean = np.mean(y)
y_std = np.std(y)
y_scaled = (y - y_mean)/y_std

def lasso_regression(X, y, lambda_, num_iters=1000, alpha=0.0005):
    m, n = X.shape
    theta = np.zeros(n)
    
    for i in range(num_iters):
        error = X @ theta - y
        gradient = (X.T @ error)/2 + (lambda_ * np.sign(theta))
        theta -= alpha * gradient
        # if np.linalg.norm(gradient, ord=2) < 0.001:
        #     break
    
    return theta

num_folds = 5
fold_size = len(X_scaled) // num_folds

best_lambda = None
best_rmse = float('inf')

#lambda_space = np.linspace(start = 0.001, stop = 1, endpoint = False, num = 50)
lambda_space = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50]
rmse_space = []
std_rmse = []
#lambda_space = np.linspace(0.05, 10, 199)
    
for lambda_ in lambda_space:
    rmse_list = []
    
    for fold in range(num_folds):
        start = fold * fold_size
        end = (fold + 1) * fold_size

        X_test_fold = X_scaled[start:end]
        y_test_fold = y[start:end]

        X_train_fold = np.concatenate([X_scaled[:start], X_scaled[end:]])
        y_train_fold = np.concatenate([y_scaled[:start], y_scaled[end:]])

        theta = lasso_regression(X_train_fold, y_train_fold, lambda_)
        y_pred = X_test_fold @ theta
        
        y_rmse = (y_pred*y_std) + y_mean
        rmse = np.sqrt(np.mean((y_rmse - y_test_fold)**2))
        rmse_list.append(rmse)
    
    avg_rmse = np.mean(rmse_list)
    std_5fold = np.std(rmse_list)
    
    rmse_space.append(avg_rmse)
    std_rmse.append(std_5fold)
    
    if avg_rmse < best_rmse:
        best_rmse = avg_rmse
        best_lambda = lambda_

final_theta = lasso_regression(X_scaled, y_scaled, best_lambda)
print(f"Best Lambda: {best_lambda}")
print(f"Best RMSE: {best_rmse}")

import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10,8))
plt.plot(lambda_space, rmse_space, linestyle='-')
plt.xlabel('lambda')
plt.ylabel('RMSE')
plt.show()






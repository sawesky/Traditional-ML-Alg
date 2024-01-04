import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = -1/m * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
    return cost

def compute_likelihood(X, y, theta):
    likelihood = np.exp(-compute_cost(X, y, theta))
    return likelihood

def compute_gradient(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    gradient = 1/m * X.T @ (h - y)
    return gradient

def predict(X, theta):
    probabilities = sigmoid(X @ theta)
    predictions = (probabilities >= 0.5).astype(int)
    return predictions


def standardize_data(X):
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    standardized_X = (X - mean_X) / std_X
    return standardized_X, mean_X, std_X

def stochastic_gradient_descent(X, y, theta, learning_rate, epochs, batch_size):
    m = len(y)
    costs = []
    likelihoods = []
    for epoch in range(epochs):
        for i in range(0, m, batch_size):
            X_mini_batch = X[i:i+batch_size]
            y_mini_batch = y[i:i+batch_size]
            gradient = compute_gradient(X_mini_batch, y_mini_batch, theta)
            #learning_rate = 0.001 if epoch == 0 else 0.001
            theta = theta - learning_rate * gradient
            cost = compute_cost(X, y, theta)
            likelihood = compute_likelihood(X, y, theta)
            costs.append(cost)
            likelihoods.append(likelihood)
    return theta, costs, likelihoods

data = np.loadtxt('multiclass_data.csv', delimiter=',')  

X = data[:, :-1]  
y_values = data[:, -1]  

X, mean_X, std_X = standardize_data(X)
X = np.hstack((np.ones((X.shape[0], 1)), X))

X_train, X_test, y_train, y_test = train_test_split(X, y_values, test_size=0.2, random_state=42)

num_classes = len(np.unique(y_train))
y_one_hot = np.eye(num_classes)[y_train.astype(int)]
theta_initial = np.zeros((X.shape[1], num_classes))

theta_all = []
costs_all = []
likelihoods_all = []
b_size = 2
for class_label in range(num_classes):
    y_class = y_one_hot[:, class_label].reshape(-1, 1)
    theta, costs, likelihoods = stochastic_gradient_descent(X_train, y_class, theta_initial[:, class_label].reshape(-1, 1),
                                               learning_rate=0.001, epochs=50, batch_size=b_size)
    theta_all.append(theta)
    costs_all.append(costs)
    likelihoods_all.append(likelihoods)

    #predictions = predict(X, theta)
    #print(f"Class {class_label + 1} - Predictions: {predictions.flatten()}")
    #print(f"Class {class_label + 1} - True Labels: {y_class.flatten()}")

accuracies = []
for class_label in range(num_classes):
    y_class = y_one_hot[:, class_label].reshape(-1, 1)
    predictions = predict(X_train, theta_all[class_label])
    accuracy_class = np.sum(predictions == y_class) / len(y_class)
    accuracies.append(accuracy_class)
    
overall_accuracy = np.mean(accuracies)
print(f"training acc: {overall_accuracy * 100:.2f}%")


# for class_label in range(num_classes):
#     plt.plot(np.squeeze(likelihoods_all[class_label]), label=f'Class {class_label + 1}')

# plt.xlabel('Mini-Batch')
# plt.ylabel('Likelihood')
# plt.legend()
# plt.show()
xlen = len(costs_all[0])
xaxis = np.arange(xlen)
for i in range(xlen):
    xaxis[i] = xaxis[i] * b_size
    
for class_label in range(num_classes):
    plt.plot(xaxis, np.squeeze(costs_all[class_label]), label=f'class {class_label}')

plt.xlabel('mini-batch')
plt.ylabel('negative log likelihood')
plt.title('alpha = 0.001, mini-batch size = 2')
plt.legend()
plt.show()

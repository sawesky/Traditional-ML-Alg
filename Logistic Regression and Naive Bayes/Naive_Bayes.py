import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm


data = np.loadtxt('multiclass_data.csv', delimiter=',')  
X = data[:, :-1]  
y = data[:, -1]   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def estimate_parameters(X, y):
    parameters = {}
    classes = np.unique(y)
    for class_label in classes:
        class_mask = (y == class_label)
        class_data = X[class_mask]
        class_mean = np.mean(class_data, axis=0)
        class_std = np.std(class_data, axis=0)
        parameters[class_label] = {'mean': class_mean, 'std': class_std}
    return parameters

def calculate_likelihood(x, mean, std):
    return norm.pdf(x, loc=mean, scale=std)

def predict(parameters, sample):
    classes = list(parameters.keys())
    probabilities = []
    for class_label in classes:
        class_params = parameters[class_label]
        class_likelihoods = calculate_likelihood(sample, class_params['mean'], class_params['std'])
        class_probability = np.prod(class_likelihoods)
        probabilities.append(class_probability)
    predicted_class = classes[np.argmax(probabilities)]
    return predicted_class

def predict_all(parameters, samples):
    return np.array([predict(parameters, sample) for sample in samples])

parameters = estimate_parameters(X_train, y_train)
y_pred = predict_all(parameters, X_train)
accuracy = accuracy_score(y_train, y_pred)
print(f"train acc: {accuracy * 100:.2f}%")

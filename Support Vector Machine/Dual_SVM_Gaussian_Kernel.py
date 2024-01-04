import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = np.genfromtxt('svmData.csv', delimiter=',')
X = data[:, :2]
y = data[:, 2]
C_vals = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
sigma_vals = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def vec_distance(X, Z):
    
    X = np.array(X)
    Z = np.array(Z)
    m, n = X.shape
    
    distances = np.zeros((m, m))
    
    for i in range(m):
        for j in range(m):
            diff = np.sum((X[i, :] - Z[j, :])**2)
            distances[i, j] = np.sqrt(diff)
            
    return distances

def vec_distance_pred(X, Z):
    
    X = np.array(X)
    Z = np.array(Z)
    distances = np.sqrt((Z[0] - X[0])**2 + (Z[1] - X[1])**2)        
    return distances

def gaussian_kernel(X, Z, sigma):
    
    distances = vec_distance(X, Z)
    K = np.exp(-distances**2 / (2 * sigma**2))
    return K

def gaussian_kernel_pred(X, Z, sigma):
    
    distances = vec_distance_pred(X, Z)
    K = np.exp(-distances**2 / (2 * sigma**2))
    return K

def train_svm_dual(X, y, C, sigma):
    
    m, n = X.shape
    K = gaussian_kernel(X, X, sigma)
    P = matrix(np.outer(y, y) * K)
    q = matrix(-1*np.ones((m, 1)))
    A = matrix(y.reshape(1, -1), (1, m), 'd')
    b = matrix(np.zeros(1))
    G = matrix(np.vstack((-1*np.eye(m), np.eye(m))))
    h = matrix(np.hstack((np.zeros(m), C*np.ones(m))))
    
    sol = solvers.qp(P, q, G, h, A, b)
    
    alpha = np.array(sol['x']).flatten()
    sv_indices = np.where(alpha > 1e-5)[0]
    sv_alphas = alpha[sv_indices]
    sv_X = X[sv_indices, :]
    sv_y = y[sv_indices]
    

    weights = np.sum((sv_alphas * sv_y)[:, np.newaxis] * sv_X, axis=0)
    bias = np.mean(sv_y - np.sum(gaussian_kernel(sv_X, sv_X, sigma) * sv_alphas * sv_y, axis=0))
    return weights, bias, sv_X, sv_y, sv_alphas

def decision_function(X, y, alphas, x_test, b, sigma):
    decision = 0
    for i in range(len(alphas)):
        decision += alphas[i] * y[i] * gaussian_kernel_pred(X[i], x_test, sigma)
    return decision + b

def accuracy(X, y, alphas, b, x_test, y_test, sigma):
    
    m, n = x_test.shape
    decisions = np.zeros((m, 1))
    for i in range(x_test.shape[0]):
        dec_x_test = x_test[i, :]
        out = decision_function(X, y, alphas, dec_x_test, b, sigma)
        out = np.sign(out)
        decisions[i] = out
    decisions = np.squeeze(decisions)
    correct_pred = np.sum(decisions == y_test)
    acc = correct_pred/len(y_test)
    return acc



def plot_decision_boundary(X, y, full_X, full_y, alphas, b, sigma=1.0):
    x_min, x_max = full_X[:, 0].min() - 0.2, full_X[:, 0].max() + 0.2
    y_min, y_max = full_X[:, 1].min() - 0.2, full_X[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    Z = np.zeros(xx.shape)
    
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            x_test = np.array([xx[i, j], yy[i, j]])
            Z[i, j] = decision_function(X, y, alphas, x_test, b, sigma)
    
    plt.contourf(xx, yy, Z, levels = 1, alpha=1, cmap=plt.cm.summer)
    plt.scatter(full_X[:, 0], full_X[:, 1], c=full_y, cmap=plt.cm.summer, edgecolors='k')
    
    supp_vecs = X[:5, :2]
    plt.scatter(supp_vecs[:, 0], supp_vecs[:, 1], s=100, facecolors='none', edgecolors='k', linewidths=1.5)
    
    supp_alphas = alphas[:5].round(2)
    for sv in supp_vecs:
        idx = np.where((X == sv).all(axis=1))[0][0]
        plt.annotate(f'{supp_alphas[idx]}', sv, textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.title('SVM dualni, Gausovski kernel')
    plt.show()

def plot_C(X, y, C_vals, sigma_vals):
    
    C_length = len(C_vals)
    sigma_length = len(sigma_vals)
    train_accuracies = np.zeros((C_length, sigma_length))
    val_accuracies = np.zeros((C_length, sigma_length))

    for i in range(C_length):
        for j in range(sigma_length):
            
            _, b, sv_X, sv_Y, sv_alphas = train_svm_dual(X_train, y_train, C_vals[i], sigma_vals[j])
            train_acc = accuracy(sv_X, sv_Y, sv_alphas, b, X_train, y_train, sigma_vals[j])
            train_accuracies[i, j]= train_acc
            val_acc = accuracy(sv_X, sv_Y, sv_alphas, b, X_val, y_val, sigma_vals[j])
            val_accuracies[i, j] = val_acc

    xx, yy = np.meshgrid(C_vals, sigma_vals)
    plt.figure()
    vmin=0.50
    vmax=1.00
    contour = plt.contourf(xx, yy, train_accuracies.T, levels=199, alpha = 1, vmin=0.50, vmax=1.00)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('C')
    plt.ylabel('sigma')
    plt.title('Train Accuracy(C, sigma)')
    cbar = plt.colorbar(contour, ticks=np.linspace(vmin, vmax, 5))
    plt.show()
    
    plt.figure()
    contour = plt.contourf(xx, yy, val_accuracies.T, levels=199, alpha = 1, vmin=0.50, vmax=1.00)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('C')
    plt.ylabel('sigma')
    plt.title('Val Accuracy(C, sigma)')
    plt.colorbar(contour, ticks=np.linspace(vmin, vmax, 5))  
    plt.show()
    return train_accuracies, val_accuracies


sigma = 1
C = 100
# sigma = 0.2
# C = 0.00001
w, b, supp_vectors, supp_vectors_c, alpha = train_svm_dual(X, y, C = C, sigma=sigma)
plot_decision_boundary(supp_vectors, supp_vectors_c, X, y, alpha, b, sigma=sigma)
trac, valac = plot_C(supp_vectors, supp_vectors_c, C_vals, sigma_vals)

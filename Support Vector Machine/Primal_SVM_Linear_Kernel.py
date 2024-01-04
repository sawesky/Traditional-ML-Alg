import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split

data = np.genfromtxt('svmData.csv', delimiter=',')
X = data[:,:2]
y = data[:,2]
C_vals = np.array([0.1, 0.5, 1, 2, 5, 10, 50, 100, 1000])
C_vals = np.linspace(0.5, 5, 100)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def hinge_loss(X, y, w, b):
    
    m, n = X.shape
    inner = np.dot(X, w) + b*np.ones((m, 1))
    y = np.resize(y, (m, 1))
    g_margin = y*inner
    return np.maximum(1 - g_margin, 0)

def accuracy(X, y, w, b):

    predictions = np.squeeze(np.sign(np.dot(X, w) + b))
    correct_pred = np.sum(predictions == y)
    acc = correct_pred/len(y)*100
    return acc
    
def train_svm_qp(X, y, C):
    
    m, n = X.shape
    P = matrix(np.vstack((np.hstack((np.identity(2),np.zeros((2, m + 1)))),
                         np.zeros((m + 1, m + 3)))))
    q = matrix(np.vstack((np.zeros((2, 1)), C*np.ones((m, 1)), np.zeros((1, 1)))))
    
    x1prod = np.reshape(-y*X[:, 0], newshape=(m, 1))
    x2prod = np.reshape(-y*X[:, 1], newshape=(m, 1))
    G = matrix(np.vstack((np.hstack((x1prod, x2prod, -1*np.identity(m), np.reshape(-y, (m, 1)))),
                          np.hstack((np.zeros((m, 2)), -1*np.identity(m), np.zeros((m, 1)))))))
    h = matrix(np.vstack((-1*np.ones((m, 1)), np.zeros((m, 1)))))
    
    sol = solvers.qp(P, q, G, h)
    weights = np.array(sol['x'][:2])
    bias = sol['x'][-1]
    
    inner = np.dot(X, weights) + bias*np.ones((m, 1))
    y = np.resize(y, (m, 1))
    g_margin = y*inner
    sv_indices = np.where(g_margin < 1)[0]
    supp_vectors = X[sv_indices, :]
    supp_vectors_c = y[sv_indices]
    
    return weights, bias, supp_vectors, supp_vectors_c 
    

def plot_svm(X, y, w, b, supp_vectors):
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.summer, edgecolors='k', marker='o')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    decision_boundary = xy.dot(w) + b
    decision_boundary = decision_boundary.reshape(xx.shape)
    
    supp_vectors = supp_vectors[:5, :2]
    plt.scatter(supp_vectors[:, 0], supp_vectors[:, 1], s=100, facecolors='none', edgecolors='k', linewidths=1.5)
    
    hinge_losses = hinge_loss(X, y, w, b).round(2)
    for sv in supp_vectors:
        idx = np.where((X == sv).all(axis=1))[0][0]
        plt.annotate(f'{hinge_losses[idx][0]}', sv, textcoords="offset points", xytext=(0,10), ha='center')

    plt.contour(xx, yy, decision_boundary, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('SVM primalni')
    plt.show()


def plot_C(X, y, C_vals):
    
    train_accuracies = []
    val_accuracies = []
    max_train_acc = 0
    max_acc_C = 0
    for C in C_vals:
        w, b, _, _ = train_svm_qp(X_train, y_train, C)
        train_acc = accuracy(X_train, y_train, w, b)
        if train_acc > max_train_acc:
            max_train_acc = train_acc
            max_acc_C = C
        train_accuracies.append(train_acc)
        val_acc = accuracy(X_val, y_val, w, b)
        val_accuracies.append(val_acc)

    print(max_acc_C)
    plt.plot(C_vals, train_accuracies, label='Training', marker='o')
    plt.plot(C_vals, val_accuracies, label='Validation', marker='o')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy(C)')
    plt.show()

w, b, supp_vectors, supp_vectors_c = train_svm_qp(X, y, C = 1.045)
plot_svm(X, y, w, b, supp_vectors)

plot_C(X, y, C_vals)
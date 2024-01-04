import numpy as np
import matplotlib.pyplot as plt
import warnings
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

#%%
warnings.filterwarnings("ignore")

data = np.genfromtxt('data_2.csv', delimiter=',')
X = data[:, :6]
y = data[:, 6]

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

param_grid_rf = {
    'n_estimators': [10, 20, 30, 40, 50, 60],
    'max_depth': [1, 2, 5, 10, 15, 20],
    'max_features': [1, 2, 3, 4, 5, 6]
}

rf_classifier = RandomForestClassifier()
#f1_scorer = make_scorer(f1_score)
grid_search_rf = GridSearchCV(rf_classifier, param_grid_rf, scoring='accuracy', cv=5, return_train_score=True)
grid_search_rf.fit(X_train, y_train)
best_params_rf = grid_search_rf.best_params_
print("Best Parameters for Random Forest:", best_params_rf)

grid_values = grid_search_rf.cv_results_['params']
test_scores = grid_search_rf.cv_results_['mean_test_score']
train_scores = grid_search_rf.cv_results_['mean_train_score']

n_estimators_values = np.array([x['n_estimators'] for x in grid_values], dtype=int)
max_depth_values = np.array([x['max_depth'] if x['max_depth'] is not None else -1 for x in grid_values], dtype=int)
max_features_values = np.array([x['max_features'] for x in grid_values], dtype=int)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=10, azim=18)

scatter = ax.scatter(n_estimators_values, max_depth_values, max_features_values, c=test_scores, cmap='viridis', s=100, alpha = 1)


ax.set_xlabel('broj stabala')
ax.set_ylabel('max dubina')
ax.set_zlabel('broj prediktora')
ax.set_title('Grid Search RF Val')


cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
cbar.set_label('mean val accuracy')

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=10, azim=18)

scatter = ax.scatter(n_estimators_values, max_depth_values, max_features_values, c=train_scores, cmap='viridis', s=100, alpha = 1)

ax.set_xlabel('broj stabala')
ax.set_ylabel('max dubina')
ax.set_zlabel('broj prediktora')
ax.set_title('Grid Search RF Train')

cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
cbar.set_label('Mean train accuracy')

plt.show()
#%%

data = np.genfromtxt('data_2.csv', delimiter=',')
X = data[:, :6]
y = data[:, 6]

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

param_grid_gb = {
    'n_estimators': [10, 20, 30, 40, 50, 60],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
    'max_depth': [1, 2, 5, 10, 15, 20]
}

gb_classifier = GradientBoostingClassifier()

grid_search_gb = GridSearchCV(gb_classifier, param_grid_gb, cv=5)
grid_search_gb.fit(X_train, y_train)
best_params_gb = grid_search_gb.best_params_
print("Best Parameters for Gradient Boosting:", best_params_gb)

grid_values = grid_search_gb.cv_results_['params']
test_scores = grid_search_gb.cv_results_['mean_test_score']

n_estimators_values = np.array([x['n_estimators'] for x in grid_values], dtype=int)
max_depth_values = np.array([x['max_depth'] if x['max_depth'] is not None else -1 for x in grid_values], dtype=int)
learning_rate_values = np.array([x['learning_rate'] for x in grid_values], dtype=float)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=10, azim=18)
scatter = ax.scatter(n_estimators_values, max_depth_values, learning_rate_values, c=test_scores, cmap='viridis', s=100, alpha = 1)

ax.set_xlabel('velicina ansambla')
ax.set_ylabel('max dubina')
ax.set_zlabel('learning rate')
ax.set_title('Grid Search GB Val')

cbar = fig.colorbar(scatter, ax=ax, orientation='vertical')
cbar.set_label('mean val accuracy')

#%%


data = np.genfromtxt('data_2.csv', delimiter=',')
X = data[:, :6]
y = data[:, 6]

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

#best_rf = RandomForestClassifier(**best_params_rf)
best_rf = RandomForestClassifier(max_depth=5, max_features=5, n_estimators=60)
best_rf.fit(X_train, y_train)

feature_importances_rf = best_rf.feature_importances_
plt.figure(figsize=(13, 4))
plt.bar(range(len(feature_importances_rf)), feature_importances_rf)
plt.xticks(range(len(feature_importances_rf)))
plt.title('RF znacajnost prediktora')
plt.show()




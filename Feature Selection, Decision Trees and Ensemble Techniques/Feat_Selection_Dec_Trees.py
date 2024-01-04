import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

data = np.genfromtxt('data_1.csv', delimiter=',')
X = data[:, :13]
y = data[:, 13]

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

correlations = []
for i in range(X_standardized.shape[1]):
    correlation = np.corrcoef(X_standardized[:, i], y)
    correlation = correlation[0, 1]
    correlations.append((i, correlation))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("corr")
for i, (predictor_index, correlation) in enumerate(correlations):
    print(f"{i + 1}. pred {predictor_index + 1}: {correlation:.4f}")


ind, accs = zip(*correlations)
ind = list(ind)
accs = list(accs)
plt.figure(figsize=(8, 5))
plt.title("Korelacije")
plt.axhline(y=0, color='black', linewidth=1)
plt.bar(range(len(ind)), list(map(float, accs)))
plt.xticks(range(len(ind)), ind)

num_features = X_standardized.shape[1]
rfe_out = []
subset = set(range(X_standardized.shape[1]))
n_f = num_features
start_score = np.mean(cross_val_score(LogisticRegression(random_state=42), X, y, cv=5))
all_delta_score = []
for i in range(n_f - 1):
    
    rfe_ranking = []
    for x in subset:
        features_without_x = [j for j in subset if j != x]
        score_without_x = np.mean(cross_val_score(LogisticRegression(random_state=42), X[:, features_without_x], y, cv=5))
        rfe_ranking.append((x, score_without_x))

    
    rfe_ranking.sort(key=lambda x: x[1], reverse=True)
    
    curr_score = rfe_ranking[0][1]
    all_delta_score.append((rfe_ranking[0][0], curr_score - start_score))
    start_score = curr_score
    
    rfe_out.append(rfe_ranking[0])
    pop_one = rfe_ranking[0][0]
    subset.discard(pop_one)

ind, accs = zip(*rfe_out)
ind = list(ind)
ind.insert(0, 'start')
start_score = np.mean(cross_val_score(LogisticRegression(random_state=42), X, y, cv=5))
accs = list(accs)
accs.insert(0, start_score)
plt.figure(figsize=(8, 5))
plt.title("Accuracy posle izbacivanja")
plt.bar(range(len(ind)), list(map(float, accs)))
plt.xticks(range(len(ind)), ind)
plt.ylim(0.9, 1.0)
# print("WrapLog")
# for i, predictor_index in rfe_out:
#     print(f"{i} pred {predictor_index}")

#%%

predictor1_index = 12
predictor2_index = 0

X = X[:, [predictor1_index, predictor2_index]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def train_and_plot_decision_tree(max_depth, figure_title):

    fig, ax = plt.subplots(figsize=(6, 4.5))

    dt_classifier = DecisionTreeClassifier(max_depth=max_depth)
    dt_classifier.fit(X_train, y_train)

    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = dt_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.8)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='o', s=50, linewidth=1, label='Trening')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', marker='s', s=50, linewidth=1, label='Test')

    ax.set_title(f'{figure_title}')
    ax.set_xlabel(f'prediktor {predictor1_index}')
    ax.set_ylabel(f'prediktor {predictor2_index}')
    ax.legend()

    plt.tight_layout()
    plt.show()

train_and_plot_decision_tree(max_depth=1, figure_title='Podobucavanje (panj)')
train_and_plot_decision_tree(max_depth=8, figure_title='Preobucavanje (dubina 8)')
train_and_plot_decision_tree(max_depth=2, figure_title='Balansirano (dubina 2)')

#%%

from sklearn.model_selection import cross_validate

depths = range(1, 16)
scoring_metrics = ['accuracy']
cv_results = {metric: {'train_mean': [], 'train_std': [], 'test_mean': [], 'test_std': []} for metric in scoring_metrics}

for depth in depths:
    dt_classifier = DecisionTreeClassifier(max_depth=depth)
    scores = cross_validate(dt_classifier, X, y, cv=5, scoring=scoring_metrics, return_train_score=True)
    
    for metric in scoring_metrics:
        cv_results[metric]['train_mean'].append(np.mean(scores[f'train_{metric}']))
        cv_results[metric]['train_std'].append(np.std(scores[f'train_{metric}']))
        cv_results[metric]['test_mean'].append(np.mean(scores[f'test_{metric}']))
        cv_results[metric]['test_std'].append(np.std(scores[f'test_{metric}']))

plt.figure(figsize=(13, 4))

for metric in scoring_metrics:
    plt.plot(depths, cv_results[metric]['train_mean'], 'o-', label=f'Trening')
    plt.fill_between(depths, np.array(cv_results[metric]['train_mean']) - np.array(cv_results[metric]['train_std']),
                     np.array(cv_results[metric]['train_mean']) + np.array(cv_results[metric]['train_std']),
                     alpha=0.2)
    
    plt.plot(depths, cv_results[metric]['test_mean'], 's-', label=f'Validacioni', color='purple')
    plt.fill_between(depths, np.array(cv_results[metric]['test_mean']) - np.array(cv_results[metric]['test_std']),
                     np.array(cv_results[metric]['test_mean']) + np.array(cv_results[metric]['test_std']),
                     alpha=0.2, color = 'purple')

plt.title('Zavisnost krosvalidacionog skora od dubine stabla')
plt.xlabel('max_depth')
plt.ylabel('acc')
plt.legend()
plt.show()
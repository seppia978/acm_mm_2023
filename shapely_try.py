import random
import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

# Load the dataset
features, target = load_breast_cancer(return_X_y=True, as_frame=True)

# Make a train/test split using 30% test size
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.30, random_state=2022
)

# prepare a model
f = make_pipeline(StandardScaler(), LogisticRegression())

# fit the model
f.fit(X_train, y_train)

sample_idx = 0
x = X_test.iloc[sample_idx]

# calculate the shaply value for feature j
j = 15
M = 1000
n_features = len(x)
marginal_contributions = []
feature_idxs = list(range(n_features))
feature_idxs.remove(j)
for _ in range(M):
    z = X_train.sample(1).values[0]
    x_idx = random.sample(feature_idxs,
                          min(max(int(0.2 * n_features), random.choice(feature_idxs)), int(0.8 * n_features)))
    z_idx = [idx for idx in feature_idxs if idx not in x_idx]

    # construct two new instances
    x_plus_j = np.ones(x.shape)
    x_plus_j[x_idx + [j]] = x[x_idx + [j]]
    x_plus_j[z_idx] = z[z_idx]

    # x_plus_j = np.array([x[i] if i in x_idx + [j] else z[i] for i in range(n_features)])
    # x_minus_j = np.array([z[i] if i in z_idx + [j] else x[i] for i in range(n_features)])
    x_minus_j = np.ones(z.shape)
    x_minus_j[z_idx + [j]] = z[z_idx + [j]]
    x_minus_j[x_idx] = x[x_idx]
    # calculate marginal contribution
    marginal_contribution = f.predict_proba(x_plus_j.reshape(1, -1))[0][1] - \
                            f.predict_proba(x_minus_j.reshape(1, -1))[0][1]
    marginal_contributions.append(marginal_contribution)

phi_j_x = sum(marginal_contributions) / len(marginal_contributions)  # our shaply value
print(f"Shaply value for feature j: {phi_j_x:.5}")

import shap

# explain the model's predictions using SHAP
explainer = shap.KernelExplainer(f.predict_proba, X_train)
shap_values = explainer.shap_values(X_test.iloc[sample_idx,:])

print(f"Shaply value calulated from shap: {shap_values[1][j]:.5}")

import pandas as pd
import autosklearn.regression
import sklearn.datasets
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data_path = "cal_housing.csv"
dependent_var = "median_house_value"
t = 60*30

df = pd.read_csv(data_path)
#print(df.head())
X = df.drop(columns=[dependent_var])
y = df[dependent_var]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=t) 
automl.fit(X_train, y_train)
print(automl.leaderboard())

train_predictions = automl.predict(X_train)
print("Train R2 score:", sklearn.metrics.r2_score(y_train, train_predictions))
test_predictions = automl.predict(X_test)
print("Test R2 score:", sklearn.metrics.r2_score(y_test, test_predictions))

plt.scatter(train_predictions, y_train, label="Train samples", c="#d95f02")
plt.scatter(test_predictions, y_test, label="Test samples", c="#7570b3")
plt.xlabel("Predicted value")
plt.ylabel("True value")
plt.legend()

min_val = min(min(train_predictions), min(test_predictions), min(y_train), min(y_test))
max_val = max(max(train_predictions), max(test_predictions), max(y_train), max(y_test))
plt.plot([min_val, max_val], [min_val, max_val], c="k", zorder=0)
plt.gca().set_aspect('equal', adjustable='box')
plt.autoscale(enable=True, axis='both', tight=True)

plt.tight_layout()

plt.savefig("plot_"  + str(t) + "t.png")

plt.show()
import pandas as pd
import autosklearn.regression
import sklearn.datasets
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("winequality-red.csv")
print(df.head())

X = df.drop(columns=['quality'])
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

t = 60*15
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
plt.plot([0, 10], [0, 10], c="k", zorder=0)
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.tight_layout()
plt.show()
plt.savefig("plot_"  + str(t) + "t.png")
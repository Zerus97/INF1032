import pandas as pd
from sklearn import linear_model

df = pd.read_csv("winequality-red.csv")

X = [] # Guarda as variáveis independentes
y = [] # Guarda as variáveis dependentes

for column in df.columns:
    if column != "quality":
        X.append(column)
    else:
        y.append(column)

X = df[X]
y = df[y]

regr = linear_model.LinearRegression() # Cria o objeto de regressão linear
regr.fit(X.values, y.values) # Pega os valores das variáveis dependentes e independentes e preenche o objeto de regressão com dados que deescrevem a relação das duas



teste = df.iloc[801]
resposta = teste["quality"]
teste = teste.drop(["quality"])
prediction = regr.predict([teste.values])
print(str(resposta) + " X "+ str(prediction))
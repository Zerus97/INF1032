import pandas as pd
from sklearn import linear_model
from random import randint

df = pd.read_csv("winequality-red.csv") # Lê csv

X = [] # Guarda as variáveis independentes
y = [] # Guarda as variáveis dependentes

for column in df.columns: # Separa as variáveis dependentes e independentes
    if column != "quality":
        X.append(column)
    else:
        y.append(column)


X = df[X].head(1600) # Quantidade de dados a serem considerados
y = df[y].head(1600) # Quantidade de dados a serem considerados

regr = linear_model.LinearRegression() # Cria o objeto de regressão linear
regr.fit(X.values, y.values) # Pega os valores das variáveis dependentes e independentes e preenche o objeto de regressão com dados que deescrevem a relação das duas



teste = df.iloc[randint(0, 1599)] # Escolha uma linha aleatória da base para fazer a previsão)
resposta = teste["quality"] # Separa a resposta
teste = teste.drop(["quality"]) # Exclui a coluna da variável dapendente
prediction = regr.predict([teste.values]) # Passa os valores para o modelo
print("Valor esperado: " + str(resposta) + "\nValor encontrado: "+ str((prediction[0][0])) + "\nValor com round: " + str(round(prediction[0][0])))

df_corr = df.corr() # Verifica a relação entre as variáveis
print(df_corr["quality"]) # Mostra quais variáveis tem mais peso na determinação da qualidade
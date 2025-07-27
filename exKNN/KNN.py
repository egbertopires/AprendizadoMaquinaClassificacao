# Importação da biblioteca
from sklearn.neighbors import KNeighborsClassifier

from ArquivoTratado.censusDF import (
    X_dataCensusDF_treino, Y_dataCensusDF_treino,
    X_dataCensusDF_teste, Y_dataCensusDF_teste)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

listaImpares = []

for numeros in range(2, 99 + 1):
    indicador = []
    if numeros % 2 != 0:
        knn_censusDF = KNeighborsClassifier(
            metric='euclidean',
            n_neighbors=numeros)

        knn_censusDF.fit(
            X_dataCensusDF_treino,
            Y_dataCensusDF_treino)

        Y_previsto = knn_censusDF.predict(X_dataCensusDF_teste)

        acuracia = accuracy_score(Y_dataCensusDF_teste,Y_previsto)*100
        classificador = classification_report(Y_dataCensusDF_teste,Y_previsto)
        matrizConfusao = confusion_matrix(Y_dataCensusDF_teste,Y_previsto)

        indicador = [acuracia, classificador, matrizConfusao, numeros, knn_censusDF]

        listaImpares.append(indicador)

listaImpares.sort(key=lambda x: x[0], reverse=True)
melhor_resultado = listaImpares[0]
melhor_modelo = melhor_resultado[4]

print(f'K = {melhor_resultado[3]}'
      f'\nAcurácia KNN: {melhor_resultado[0]:.2f}%\n')
print(f'Relatório de classificação (KNN):\n{melhor_resultado[1]}\n')
print(f'Matriz de confusão (KNN):\n{melhor_resultado[2]}')



from sklearn.ensemble import RandomForestClassifier

from ArquivoTratado.censusDF import (
    X_dataCensusDF_treino, X_dataCensusDF_teste,
    Y_dataCensusDF_treino, Y_dataCensusDF_teste)

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

# Criação da instância:
censusRandomForest = RandomForestClassifier(n_estimators=57,criterion='entropy', random_state = 0,max_depth=12)

# Treinando o modelo de aprendizagem:
censusRandomForest.fit(X_dataCensusDF_treino,Y_dataCensusDF_treino)

# Testando o modelo de aprendizagem:
previsaoRandomForest = censusRandomForest.predict(X_dataCensusDF_teste)

# Relatório:
print(f'\nAcurácia (Random Forest): '
      f'{(accuracy_score(Y_dataCensusDF_teste,previsaoRandomForest)*100):.2f}%\n')
print(f'Relatório de classificação (Random Forest):'
      f'\n{classification_report(Y_dataCensusDF_teste,previsaoRandomForest)}\n')
print(f'Matriz de confusão (Random Forest): '
      f'\n{confusion_matrix(Y_dataCensusDF_teste,previsaoRandomForest)}')

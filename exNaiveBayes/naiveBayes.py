from ArquivoTratado.censusDF import (
    X_dataCensusDF_treino, Y_dataCensusDF_treino,
    X_dataCensusDF_teste, Y_dataCensusDF_teste)

# Modelo de aprendizagem de máquina Naive Bayes

# Importando a biblioteca que trabalha apenas com dados numéricos
from sklearn.naive_bayes import GaussianNB

naive_Numerico = GaussianNB()

naive_Numerico.fit(X_dataCensusDF_treino,Y_dataCensusDF_treino)

# Acurácia, Relatório de classificação e Matriz de confusão

# Importando as bibliotecas necessárias
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

Y_previsto = naive_Numerico.predict(X_dataCensusDF_teste)

print(f'\nAcurácia NAIVE: '
      f'{(accuracy_score(Y_dataCensusDF_teste,Y_previsto)*100):.2f}%\n')
print(f'Relatório de classificação (Naive):'
      f'\n{classification_report(Y_dataCensusDF_teste,Y_previsto)}\n')
print(f'Matriz de confusão (Naive): '
      f'\n{confusion_matrix(Y_dataCensusDF_teste,Y_previsto)}')


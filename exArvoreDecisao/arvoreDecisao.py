from sklearn.tree import DecisionTreeClassifier

from ArquivoTratado.censusDF import (
    X_dataCensusDF_treino, X_dataCensusDF_teste,
    Y_dataCensusDF_treino, Y_dataCensusDF_teste)

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

# Criando a instância da árvore de decisão:
arvoreDecisao = DecisionTreeClassifier(criterion='entropy',max_depth=12)
# arvoreDecisao = DecisionTreeClassifier(criterion='entropy')

# Treinando a minha árvore:
arvoreDecisao.fit(X_dataCensusDF_treino, Y_dataCensusDF_treino)

previsao = arvoreDecisao.predict(X_dataCensusDF_teste)

print(f'\nAcurácia ÁRVORE: '
      f'{(accuracy_score(Y_dataCensusDF_teste,previsao)*100):.2f}%\n')
print(f'Relatório de classificação (Árvore):'
      f'\n{classification_report(Y_dataCensusDF_teste,previsao)}\n')
print(f'Matriz de confusão (Árvore): '
      f'\n{confusion_matrix(Y_dataCensusDF_teste,previsao)}')
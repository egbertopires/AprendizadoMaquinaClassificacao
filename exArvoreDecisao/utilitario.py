from arvoreDecisao import arvoreDecisao

from sklearn.tree import DecisionTreeClassifier

# Verificando a importância das variáveis (entropia)
listaEntropia = list(arvoreDecisao.feature_importances_)
listaEntropia.sort(reverse=True)
listaEntropia = [float(x) for x in listaEntropia]
print(listaEntropia)
print(len(listaEntropia))


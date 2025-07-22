# importando a biblioteca pandas
import pandas as pd

# importando a base de dados census.csv
dataCensusDF = pd.read_csv("../arquivos/census.csv")

# retirando os espaços vazios do conjunto de dados
dataCensusDF = dataCensusDF.apply(lambda x: x.str.strip()
                          if x.dtype == "object" else x)

# (Mudando a escriata de algumas variáveis.
#  Criando a variável dicionário)
renomearColunas = {
    'final-weight': 'final_weight',
    'education-num': 'education_num',
    'marital-status': 'marital_status',
    'capital-gain': 'capital_gain',
    'capital-loos': 'capital_loos',
    'hour-per-week': 'hour_per_week',
    'native-country': 'native_country'
}

dataCensusDF = dataCensusDF.rename(columns=renomearColunas)

# Fazendo a separação entre previsores e classe
X_dataCensusDF = dataCensusDF.iloc[:,0:-1].values
Y_dataCensusDF = dataCensusDF.iloc[:,-1].values

# Codificando as variáveis categórica

# Importação de biblioteca
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer


LabelEncoder_workclass = LabelEncoder()
LabelEncoder_education = LabelEncoder()
LabelEncoder_marital_status = LabelEncoder()
LabelEncoder_occupation = LabelEncoder()
LabelEncoder_relationship = LabelEncoder()
LabelEncoder_race = LabelEncoder()
LabelEncoder_sex = LabelEncoder()
LabelEncoder_native_country = LabelEncoder()

X_dataCensusDF[:,1] = LabelEncoder_workclass.fit_transform(
    X_dataCensusDF[:,1])
X_dataCensusDF[:,3] = LabelEncoder_education.fit_transform(
    X_dataCensusDF[:,3])
X_dataCensusDF[:,5] = LabelEncoder_marital_status.fit_transform(
    X_dataCensusDF[:,5])
X_dataCensusDF[:,6] = LabelEncoder_occupation.fit_transform(
    X_dataCensusDF[:,6])
X_dataCensusDF[:,7] = LabelEncoder_relationship.fit_transform(
    X_dataCensusDF[:,7])
X_dataCensusDF[:,8] = LabelEncoder_race.fit_transform(
    X_dataCensusDF[:,8])
X_dataCensusDF[:,9] = LabelEncoder_sex.fit_transform(
    X_dataCensusDF[:,9])
X_dataCensusDF[:,13] = LabelEncoder_native_country.fit_transform(
    X_dataCensusDF[:,13])

# Normalizando os valores das variáveis categóricas codificadas e das numéricas

# Importação da biblioteca
from sklearn.preprocessing import MinMaxScaler

# Criação da instância de normaização
normalizacao_X_dataCensusDF = MinMaxScaler()

# Normalizando as variáveis previsoras
X_dataCensusDF_normalizado = normalizacao_X_dataCensusDF.fit_transform(X_dataCensusDF)

X_dataCensusDF_normalizado2 = normalizacao_X_dataCensusDF.fit_transform(X_dataCensusDF)

# Separando os dados de treino e teste.

# Importação da biblioteca
from sklearn.model_selection import train_test_split

(X_dataCensusDF_treino, X_dataCensusDF_teste, Y_dataCensusDF_treino,Y_dataCensusDF_teste) = train_test_split(
    X_dataCensusDF_normalizado,Y_dataCensusDF,test_size=0.2,random_state=100)

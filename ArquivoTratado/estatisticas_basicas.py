from ArquivoTratado.censusDF import dataCensusDF, X_dataCensusDF_normalizado
import  pandas as pd
# Configurações para exibição completa:

# Mostrar todas as colunas do DataFrame, sem limite
pd.set_option('display.max_columns', None)

# Mostrar todas as linhas do DataFrame, sem limite (cuidado com tabelas grandes)
#pd.set_option('display.max_rows', None)

# Ajustar a largura do console para não quebrar a tabela (sem limite de largura)
#pd.set_option('display.width', None)

# Mostrar o conteúdo completo de cada célula, sem cortar texto com "..."
pd.set_option('display.max_colwidth', None)


print('BASE DE DADOS CENSUS\n\n'
      'TABELA:\n'
      f'{dataCensusDF.head(2)}\n\n\n'
      f'INFORMAÇÕES GERAIS:\n')
print(dataCensusDF.info())
print(f'\n\n\nMEDIDAS ESTATÍSTICAS BÁSICAS:\n'
      f'{dataCensusDF.describe().round(2)}')



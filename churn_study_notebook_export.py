# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Churn Classification Study
# ### Saulo Accacio de Oliveira - saulo.accacio@gmail.com
#
# **Dataset File**
#
# Churn Train.xlsx - Telecom churn dataset
#
# **ATENTION**
# Todas as bibliotecas foram atalizadas para suas últimas versões por conta do pacote de metricas MAPE, inclusive a sklearn que precisaestar na versão 0.24.1.
# Para forçar a atualização utilizar os procedimentos descritos em https://anaconda.org/conda-forge/scikit-learn
# Se deseja utilizar os pacotes stardard sem esse upgrade, remover a metrica MAPE (só existe naversão 0.24)
#
# Don't forget to setup home variable to your path.
# %% [markdown]
# ## Instalação das bibliotecas utilizadas
# Obs: Essa etapa pode ser executada apenas uma vez.

# %%
get_ipython().system('pip install pandas')
get_ipython().system('pip install xlrd')
get_ipython().system('pip install openpyxl ')
get_ipython().system('pip install numpy')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')
# minimum required ver 0.24.1
get_ipython().system('pip install -U sklearn')


# %%
# bibliotecas
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format #configurando casas decimais
import numpy as np
#somente para o notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as snb
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

#variaveis glboais
home = ''
#arquivo de treino

churn_train = home + 'Churn Train.xlsx'
feat_churn_x = ['state', 'account_length', 'area_code', 'international_plan', 'voice_mail_plan', 'number_vmail_messages', 'total_day_minutes'
                'total_day_calls', 'total_day_charge','total_eve_minutes', 'total_eve_calls','total_eve_charge','total_night_minutes',
                'total_night_calls', 'total_night_charge', 'total_intl_minutes', 'total_intl_calls', 'total_intl_charge', 'number_customer_service_calls']
feat_churn_y = ['churn']

def print_status_classifier(model, classifier,  X_train, y_train, X_test, y_test, y_pred):
    print('AVALIAÇÃO PRINCIPAL')
    print('==========================')
    print('   Test Accuracy Score: %.2f' % metrics.accuracy_score(y_test, y_pred, normalize=False))

    print('')
    print('OUTROS INDICADORES')
    print(classification_report(y_test, y_pred, target_names=['Abandono', 'Não Abandono']))
    plot_confusion_matrix(classifier, X_test, y_test, cmap="Blues")
    plt.show()


# %%
# Importação do dataset
df_churn_train = pd.read_excel(churn_train, sheet_name='Sheet1')
df_churn_train.head()

# %% [markdown]
# ### Realizando a avaliação inicial dos dados
# Para realizar a etapa de preparação dos dados
# - Valores omissos
# - Diferença entre valores de features
# - Remover valores dupliados
# - Avaliar o contexto dos dados em geral

# %%
df_churn_train.info()
df_churn_train.isna().sum()


# %%
# Estatistica descritiva dos dados
df_churn_train.describe()


# %%
#novo dataset
df_churn_train.info()


# %%
#plotando um histograma das colunas para termos uma avaliação dos dados como um todo normalidade e outliers
df_churn_train.hist(bins=25, figsize=(20,15))
plt.show()


# %%
# Avaliando acorrelação entre os dados
corr_matrix = df_churn_train.corr()
corr_matrix


# %%
plt.figure(figsize = (15,15))
snb.heatmap(corr_matrix, data = df_churn_train)

# %% [markdown]
# ### Apagando duplicatas e fazendo a normalização das escalas

# %%
#apagando dados duplicados
df_churn_train.drop_duplicates(keep = False, inplace = True)

#normalizando colunas
from sklearn.preprocessing import StandardScaler

scaler = MinMaxScaler()
numerical_features = df_churn_train.select_dtypes(include = ['float64', 'int64']).columns.tolist()

df_churn_train[numerical_features] = scaler.fit_transform(df_churn_train[numerical_features])
df_churn_train[numerical_features].describe()

# %% [markdown]
# ### Tratando todas as variáveis categorias
# Para as variaveis categoricas utilizaremos a tecninca de One Hot Encoding nas colunas 'state', 'area_code' e uma codificação simples nas colunas com classificação binária (0 e 1).
#

# %%
#Codificando variáveis binárias como o e 1
df_churn_train.churn = pd.Categorical(df_churn_train.churn).codes
df_churn_train.international_plan = pd.Categorical(df_churn_train.international_plan).codes
df_churn_train.voice_mail_plan = pd.Categorical(df_churn_train.voice_mail_plan).codes

#Aplicando oneHotEncoding
df_churn_tratado = pd.get_dummies(df_churn_train, columns=['state', 'area_code'])
df_churn_tratado.head()

# %% [markdown]
# ### Separando dados de treino e teste

# %%
#separando as colunas em features e labels
def obter_treino_teste(dataset):
    X = dataset.drop(['churn'],axis = 1).values
    y = dataset['churn'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    return  X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = obter_treino_teste(df_churn_tratado)

# %% [markdown]
#
# ### Treinando os modelos

# %%
# Tratando o warning quando estoura a capacidade do gradient decendent
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore",  category = ConvergenceWarning)
from sklearn.ensemble import GradientBoostingClassifier

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier(max_depth=5)))
models.append(('GB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RF', RandomForestClassifier()))
models.append(('GPC', GaussianProcessClassifier(1.0 * RBF(1.0))))
models.append(('AB', AdaBoostClassifier()))
models.append(('GBC', GradientBoostingClassifier()))
#models.append(('LGBM', LGBMClassifier()))#n_estimators=200,learning_rate=0.11, min_child_samples=30,num_leaves=60


def rodar_modelo(models):
    results = []
    names = []
    scoring = 'accuracy'
    print('Resultado da avaiação')
    for name, model in models:
        kfold = model_selection.KFold(n_splits=5, random_state=5, shuffle=True)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results.mean())
        names.append(name)
        msg = "  %s: media %f (dp %f - max %f)" % (name, cv_results.mean(), cv_results.std(), cv_results.max())
        print(msg)
    pos_melhor_modelo = results.index(max(results))
    return models[pos_melhor_modelo]

melhor_modelo = rodar_modelo(models)
print('')
print('Melhor modelo:', melhor_modelo[1])

# %% [markdown]
# ### Avaliando a importancia das colunas

# %%
# treinando novamente o melhor modelo
def avaliar_features(melhor_modelo):
    model = melhor_modelo[1]
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    colunas = df_churn_tratado.columns

    # summarize feature importance
    for i,v in enumerate(importance):
        print('%s: %.5f' % (colunas[i],v))

    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()

avaliar_features(melhor_modelo)

# %% [markdown]
# ### Apagando as colunas com empenho reduzido e refazendo os testes
# Avaliando a relevancia das colunas, percebemos que 'area_code', 'state' tem baixissima relevancia.
#

# %%
col_colineares = ['area_code', 'state']
df_churn_tratado = df_churn_train.drop(col_colineares, axis = 1)
X_train, X_test, y_train, y_test = obter_treino_teste(df_churn_tratado)

melhor_modelo = rodar_modelo(models)
print('')
print('Melhor modelo:', melhor_modelo[1])


# %%
# Avaliando novamente
avaliar_features(melhor_modelo)

# %% [markdown]
# Com o o dataset refinado, o algoritmo **RandomForestClassifier** passa a ter um desempenho superior.
# Com isso vamos seguir para a próxima etapa, que será um ajuste fino dos hyperparametros.
# %% [markdown]
# ### Refinando os melhores modelos

# %%
from sklearn.model_selection import GridSearchCV

gsc = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid={
            'random_state': [None, 0, 1, 2, 3],
            'max_depth': [None, 1, 2, 3, 4, 15, 20],
            'n_estimators': [5, 10, 100, 150, 200],
            'max_features': ['auto', 'sqrt', 'log2', 1, 2, 3],
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 4, 6],
            'min_samples_leaf': [3,4,5]
        },
        cv=5, scoring='accuracy', verbose=0, n_jobs=-1)

grid_result = gsc.fit(X_train, y_train.ravel())
best_params = grid_result.best_params_
classifier = RandomForestClassifier(max_depth=best_params["max_depth"], random_state=best_params["random_state"],
                                   n_estimators = best_params["n_estimators"], max_features = best_params["max_features"],
                                   criterion= best_params["criterion"], min_samples_split=best_params["min_samples_split"],
                                   min_samples_leaf = best_params["min_samples_leaf"])

model = classifier.fit(X_train, y_train.ravel())
y_pred = model.predict(X_test) # avaliando o melhor estimador

print("RandomForestClassifier GIRD")
for k, v in best_params.items():
    print(' ', k, ': ',  v)

print('')

print_status_classifier(model,classifier, X_train, y_train, X_test, y_test, y_pred)

# %% [markdown]
# ### Fazer a avaliação de Overfitting
# %% [markdown]
# ### Fontes
# SKLEARN CLASSIFIERS
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
#
# Choosing a Scikit-learn Linear Regression Algorithm
# https://towardsdatascience.com/choosing-a-scikit-learn-linear-regression-algorithm-dd96b48105f5
#
# Livro: Python Data Science Handbook - VanderPlas, Jake - O'REILLY
# Disponível em: https://github.com/jakevdp/PythonDataScienceHandbook
#
# Artigos sobre Machine Learning Disponíveis em
# https://towardsdatascience.com/
# https://medium.com/
# https://kaggle.com/

# %%

#!/usr/bin/env python
# coding: utf-8

# # Análise do Airbnb (RJ)
#  Ferramenta de previsão de preço de imóvel para Pessoas Comuns (cpf)

# ---

# ## Contexto e objetivo:
# 
# - No Airbnb, qualquer pessoas que tenha um quarto ou um imóvel de qualquer tipo (apartamento, casa, chalé, pousada, etc.) pode ofertar o seu imóvel para ser alugado por diária.
# - Ao criar seu perfil de host, poderá disponibilizar e anunciar tal imóvel.
# - Neste anuúncio, será descrito o imóvel da forma mais atrativa e completa possivel, de forma a ajudar os locadores a escolherem o melhor imóvel para eles. Estas informações variam desde quantidade mínima de diárias aceitas para um contrato até quantidade de quartos e regras de cancelamento. 
# 
# - Contruir um modelo de previsão de preço que permita um host saber quanto deve cobrar pela diária do seu imóvel. 
# - Também, quanto um locador, segundo seu próprio filtro de busca, verificar qual o imóvel mais atrativo.
# 

# ## Informações extras:
# 1. A origem dos dados será [Kaggle](https://www.kaggle.com/datasets/allanbruno/airbnb-rio-de-janeiro)
# 2. A moeda padrão é **BRL** (R$).
# 3. Inspirada na solução do próprio *Allan Bruno*, mas há algumas diferenças de construção.
# 4. Análise feita por um **estudante**, logo toda ajuda será bem-vinda.
# 5. Divirta-se.
# 

# ---

# ## 1. Análise:

# ### 1.1 Importando bibliotecas

# In[1]:


import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split


# ### 1.2 Visualizando a base da dados

# In[2]:


meses = {'jan':1, 'fev':2, 'mar':3, 'abr':4, 'mai':5, 'jun':6, 'jul':7, 'ago':8, 'set':9, 'out':10, 'nov':11, 'dez':12}
caminho_bases = pathlib.Path("DataBaseAirbnb")
base_airbnb = pd.DataFrame()

for arquivo in caminho_bases.iterdir():
    mes = meses[arquivo.name[:3]]
    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))
    
    df = pd.read_csv(caminho_bases / arquivo.name)
    df['ano'] = ano
    df['mes'] = mes
    base_airbnb = base_airbnb.append(df)


# ### 1.3 Exploração e Análise e Limpeza do excesso de dados

# In[3]:


base_airbnb.head(10)


# #### 1.3.1 Nota sobre a limpeza dos dados:
# > Após analisar o conteúdo de cada coluna, decidí utilizar estas colunas, pelos seguintes motivos:
# >1. Exceto de NaN. > 300mil NaN.
# >2. Conteúdo inutilizável segundo a LGPD e/ou irrelevante para a análise.

# In[4]:


base_airbnb = base_airbnb[['host_response_time', 'host_response_rate', 'host_is_superhost', 'host_listings_count', 'latitude', 'longitude', 'property_type', 
             'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'price', 'security_deposit', 'cleaning_fee', 
             'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights', 'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 
             'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 
             'instant_bookable', 'is_business_travel_ready', 'cancellation_policy', 'ano', 'mes']]


# In[5]:


base_airbnb.isnull().sum()


# In[6]:


for col in base_airbnb:
    if base_airbnb[col].isnull().sum() > 300000:
        base_airbnb = base_airbnb.drop(col , axis = 1)
base_airbnb = base_airbnb.dropna()
base_airbnb.isnull().sum()


# #### 1.3.2 Nota sobre o tipo dos dados:
# > Analisando os tipos de dados, verifica-se a necessidade de ajusta-los, como 'price', 'extra_people'.

# In[7]:


base_airbnb['price'] = base_airbnb['price'].str.replace('$','')
base_airbnb['price'] = base_airbnb['price'].str.replace(',','')
base_airbnb['price'] = base_airbnb['price'].astype(np.float32, copy = False)


# In[8]:


base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$','')
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace(',','')
base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float32, copy = False)


# In[9]:


print(base_airbnb.dtypes)


# ### 1.4 Analise das correlações dos outliers segundo os quartis.
# 
# - verificar qual a correlação entre eles
# - Verificar dentro os valores numéricos inteiros, os números contínuos.
# - Verificar os valores "floats", os números discretos. 
# 
# >Retirando os outliers fica mais fácil a visualização dos dados em função do objetivo atual. 

# In[10]:


plt.figure(figsize = (15,8))
sns.heatmap(base_airbnb.corr(), annot = True, cmap = 'Greens')
#base_airbnb.corr()


# #### 1.4.1 Nota sobre a funcionalidade:
# - Função para desenvolver os limites máximos e mínimo usados nos gráficos.
# - Função para retirar os outliers quando houver necessidade. 
# - Caso não necessite da exclusão do outlier, será sinalizado.

# In[11]:


def lim(col):
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    amp = q3 - q1
    return q1 - 1.5 * amp, q3 + 1.5 * amp


def outliers(df, coluna):
    rows = df.shape[0]  #pegando a quantidade de linhas que há na coluna específica
    lim_inf, lim_sup = lim(df[coluna]) #pegando o valor do limite inferior e superior para definir o período que quer analisar
    df = df.loc[(df[coluna] >= lim_inf) & (df[coluna] <= lim_sup), :] #definindo o período que quer utilizar para plotar segundo o limite inferior e superior
    return df, rows - df.shape[0] #retornando o dataframe atualizado e a quantidade de linhas excluídas(outliers) 


# #### 1.4.1 Nota sobre a funcionalidade ²:
# - plotar gráficos em boxplot.
# - plotar gráficos em histograma.
# - plotar gráficos em barra.

# In[12]:


def boxplot(col):
    fig, (ax1, ax2) = plt.subplots(1, 2) #para criar 2 gráficos define-se a qtd de linhas e colunas usadas no plot. neste caso 1 linha e 2 colunas
    fig.set_size_inches(15,5) #tamanho da figura em geral
    
    sns.boxplot(x = col, ax = ax1) #gráfico com outliers
    
    ax2.set_xlim(lim(col)) #configurando a reta X e descartando os outliers
    sns.boxplot(x = col, ax= ax2) #gráfico sem outliers
    

def histogram(col):
    plt.figure(figsize = (10, 5))
    sns.displot(col, kind = 'kde')
    
def bar_plot(col):
    plt.figure(figsize = (10, 5)) 
    ax = sns.barplot(x = col.value_counts().index, y = col.value_counts())
    ax.set_xlim(lim(col))


# ### 1.5 Gráficos

# #### 1.5.1 Preço da diária

# In[13]:


# df_preço, price_row = outliers(base_airbnb, 'price')
# print(f'Total de outliers: {price_row} Linhas')
# histogram(df_preço['price'])
# boxplot(base_airbnb['price'])


# #### 1.5.2 Preço adicional por pessoas extras

# In[14]:


# df_e_people, e_people_row = outliers(base_airbnb, 'extra_people')
# print(f'Total de outliers: {e_people_row} Linhas')
# boxplot(base_airbnb['extra_people'])
# histogram(df_e_people['extra_people'])


# ####  1.5.3 Quantidade de imóveis/host

# In[15]:


df_hlc, hlc_row = outliers(base_airbnb, 'host_listings_count')
print(f'Total de outliers: {hlc_row} Linhas')
boxplot(base_airbnb['host_listings_count'])
bar_plot(df_hlc['host_listings_count'])


# #### 1.5.4 Quantidade de pessoas/imóvel

# In[16]:


df_acom, acom_row = outliers(base_airbnb, 'accommodates')
print(f'Total de outliers: {acom_row} Linhas')
boxplot(base_airbnb['accommodates'])
bar_plot(df_acom['accommodates'])


# #### 1.5.5 Quantidade de banheiros

# In[17]:


df_bath, bath_row = outliers(base_airbnb, 'bathrooms')
print(f'Total de outliers: {bath_row} Linhas')
boxplot(base_airbnb['bathrooms'])
plt.figure(figsize=(15,5))
sns.barplot(x=df_bath['bathrooms'].value_counts().index,y=df_bath['bathrooms'].value_counts())


# #### 1.5.6 Quantidade de camas e quartos.

# In[18]:


df_brooms, brooms_row = outliers(base_airbnb, 'bedrooms')
print(f'Total de outliers: {brooms_row} Linhas')
boxplot(base_airbnb['bedrooms'])
bar_plot(df_brooms['bedrooms'])


# In[19]:


df_beds, beds_row = outliers(base_airbnb, 'beds')
print(f'Total de outliers: {beds_row} Linhas')
boxplot(base_airbnb['beds'])
bar_plot(df_beds['beds'])


# #### 1.5.7 Quantidade de convidados por cliente

# In[20]:


plt.figure(figsize=(15,5))
sns.barplot(x=base_airbnb['guests_included'].value_counts().index,y=base_airbnb['guests_included'].value_counts())


# > Após a análise, verifiquei que os usuários do airbnb comumente usam o padrão(default) como preenchimento de cadastro na plataforma, como 1 guest included, logo, resolvi retirar esta feature da análise por entender que será mais benéfico ao modelo de M.L. .

# In[21]:


print('Quantidade de colunas antes do drop: {}'.format(base_airbnb.shape[1]))
base_airbnb = base_airbnb.drop('guests_included', axis = 1)
print('Quantidade de colunas após o drop: {}'.format(base_airbnb.shape[1]))


# #### 1.5.8 Quantidade mínima de noites

# In[22]:


df_min, min_row = outliers(base_airbnb, 'minimum_nights')
print(f'Total de outliers: {min_row} Linhas')
boxplot(base_airbnb['minimum_nights'])
bar_plot(df_min['minimum_nights'])


# #### 1.5.9 Quantidade máxima de noites

# In[23]:


df_max, max_row = outliers(base_airbnb, 'maximum_nights')
boxplot(base_airbnb['maximum_nights'])


# > Após a análise, verifiquei que os usuários do airbnb comumente usam o padrão(default), ou nem preenchem, como preenchimento de cadastro na plataforma, assim como no caso do guest included, logo, resolvi retirar esta feature da análise por entender que será mais benéfico ao modelo de M.L. .

# In[24]:


print('Quantidade de colunas antes do drop: {}'.format(base_airbnb.shape[1]))
base_airbnb = base_airbnb.drop('maximum_nights', axis = 1)
print('Quantidade de colunas após o drop: {}'.format(base_airbnb.shape[1]))


# #### 1.5.10 Número de reviews

# In[25]:


df_reviews, reviews_row = outliers(base_airbnb, 'number_of_reviews')
boxplot(base_airbnb['number_of_reviews'])


# > Após a análise, verifiquei que não seria apropriado utilizar a quantidade de views no modelo para o proposito, que seria focado em pessoas que não tem views na plataforma. Logo, ao usar estaria considerando pessoas com muitos views e caso eu retirasse os outliers, estaria desconsiderando pessoas com mais tempo na plataforma, mas ainda sim considerando outras não iniciante.  

# In[26]:


print('Quantidade de colunas antes do drop: {}'.format(base_airbnb.shape[1]))
base_airbnb = base_airbnb.drop('number_of_reviews', axis = 1)
print('Quantidade de colunas após o drop: {}'.format(base_airbnb.shape[1]))


# #### 1.5.11 Tipos de propriedade

# > Foi verificado que nas features onde o type era Object (string), necessitava do agrupamentos, pois os valores estavam demasiadamente distantes e diferentes. Para simplificar o aprendizado da máquina, foi feito isto. 

# In[27]:


base_airbnb['property_type'].value_counts()

plt.figure(figsize = (15, 5))
ax = sns.countplot(x = base_airbnb['property_type'])
ax.tick_params(axis = 'x', rotation = 90)


# In[28]:


table_casa = base_airbnb['property_type'].value_counts()
col_agrupadas = []

for i in table_casa.index:
    if table_casa[i] < 2200:
        col_agrupadas.append(i)
        
for i in col_agrupadas:
    base_airbnb.loc[base_airbnb['property_type'] == i, 'property_type'] = 'Outros' 


# In[29]:


print(base_airbnb['property_type'].value_counts())
plt.figure(figsize = (15,5))
grafico = sns.countplot(x = base_airbnb['property_type'])
grafico.tick_params(axis='x', rotation = 90)


# #### 1.5.12 Tipos de quarto

# In[30]:


print(base_airbnb['room_type'].value_counts())

plt.figure(figsize = (15,5))
grafico = sns.countplot(x = base_airbnb['room_type'])
grafico.tick_params(axis='x', rotation = 90)


# #### 1.5.13 Tipos de cama

# > Foi verificado que nas features onde o type era Object (string), necessitava do agrupamentos, pois os valores estavam demasiadamente distantes e diferentes. Para simplificar o aprendizado da máquina, foi feito isto. 

# In[31]:


print(base_airbnb['bed_type'].value_counts())
plt.figure(figsize = (15,5))
grafico = sns.countplot(x = base_airbnb['bed_type'])
grafico.tick_params(axis='x', rotation = 90)


# In[32]:


table_beds = base_airbnb['bed_type'].value_counts()
col_agrupadas_beds = []

for i in table_beds.index:
     if table_beds[i] < 10000:
        col_agrupadas_beds.append(i)

for i in col_agrupadas_beds:
    base_airbnb.loc[base_airbnb['bed_type'] == i, 'bed_type'] = 'Outros' 


# In[33]:


print(base_airbnb['bed_type'].value_counts())
plt.figure(figsize = (15,5))
grafico = sns.countplot(x = base_airbnb['bed_type'])
grafico.tick_params(axis='x', rotation = 90)


# #### 1.5.14 Cancelamento

# > Foi verificado que nas features onde o type era Object (string), necessitava do agrupamentos, pois os valores estavam demasiadamente distantes e diferentes. Para simplificar o aprendizado da máquina, foi feito isto. 

# In[34]:


print(base_airbnb['cancellation_policy'].value_counts())
plt.figure(figsize = (15,5))
grafico = sns.countplot(x = base_airbnb['cancellation_policy'])
grafico.tick_params(axis='x', rotation = 90)


# In[35]:


table_canc = base_airbnb['cancellation_policy'].value_counts()
col_agrupadas_canc = []

for i in table_canc.index:
     if table_canc[i] < 17000:
        col_agrupadas_canc.append(i)

for i in col_agrupadas_canc:
    base_airbnb.loc[base_airbnb['cancellation_policy'] == i, 'cancellation_policy'] = 'Strict' 


# In[36]:


print(base_airbnb['cancellation_policy'].value_counts())
plt.figure(figsize = (15,5))
grafico = sns.countplot(x = base_airbnb['cancellation_policy'])
grafico.tick_params(axis='x', rotation = 90)


# #### 1.5.15 Tipos de propriedade

# > Foi constatado nesta feature que temos uma diversas ameneties de diversas formas, logo, irei avaliar quantitativamente e não qualitativamente substituindo-a e após isso, analisar-la.

# In[37]:


print('Quantidade de colunas antes da troca: {}'.format(base_airbnb.shape[1]))
base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len) #adiciono a coluna n_amenities com os valores quantitativos.
base_airbnb = base_airbnb.drop('amenities', axis = 1) #deleto a coluna amenities com valores qualitativos.
print('Quantidade de colunas após a troca: {}'.format(base_airbnb.shape[1]))


# In[38]:


df_n_amenities, n_amenities_row = outliers(base_airbnb, 'n_amenities')
print(f'Total de outliers: {n_amenities_row} Linhas')
boxplot(base_airbnb['n_amenities'])
bar_plot(df_n_amenities['n_amenities'])


# --------------------------------------------------------------------------------------------------------------------------------------------

# ### 1.6 MAPA INTERATIVO

# #### 1.6.0 Preço por Região

# In[39]:


amostra = base_airbnb.sample(n = 90000) #usei 90.000 por ser equivalente à 10% das linhas
centro_mapa = {'lat':amostra.latitude.mean(), 'lon':amostra.longitude.mean()}
fig = px.density_mapbox(amostra,
                        lat = 'latitude',
                        lon = 'longitude',
                        z = 'price',
                        radius = 10,
                        center = centro_mapa,
                        zoom = 10,
                        mapbox_style = 'stamen-terrain'
                       )
fig.show()


# ### 1.7 Encoding
# > Será necessário criar um BACKUP para transformar as features em False(0) ou True(1) no método de variáveis Dummies (criar matrizes).

# In[40]:


backup_data = base_airbnb.copy() #backup


# In[41]:


col_tf = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready'] #features binárias

for i in col_tf:
    backup_data.loc[backup_data[i] == 't',i] = 1
    backup_data.loc[backup_data[i] == 'f',i] = 0
print(backup_data.iloc[0])


# In[42]:


col_dummies = ['property_type', 'room_type', 'bed_type', 'cancellation_policy'] #features não binárias.
backup_data = pd.get_dummies(data = backup_data, columns = col_dummies)
backup_data.head(10)


# ### 1.8 Modelos de previsão

# #### 1.8.0 Métricas de avaliação

# In[43]:


def avaliar_modelo(nome_modelo, y_teste, previsao):  #y_teste = valores reais; previsão = valores previstos pelo modelo
    r2 = r2_score(y_teste, previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo: {nome_modelo}\nR²: {r2}\nRSME: {RSME}'


# - Escolha dos modelos a serem testados
# 1. Random Forest
# 2. Linear Regression
# 3. Extra Tree

# In[44]:


modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()

modelos = {'RandomForest': modelo_rf,
          'LinearRegression': modelo_lr,
          'ExtraTrees': modelo_et,
          }
y = backup_data['price']
x = backup_data.drop('price', axis = 1)


# - Separa os dados em treino e teste + Treino do modelo

# In[45]:


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1)

for nome_modelo, modelo in modelos.items():
    #treinar
    modelo.fit(x_train, y_train)
    #testar
    previsao = modelo.predict(x_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# ### 1.9 Análise do melhor modelo para ajustes

# ###### Modelo escolhido: Extra Trees Regressor.
# - Esse foi o modelo escolhido por ter apresentado melhores resultados de R² e RSME e por não haver demasiada variação entre a velocidade de processamento de treino e previsão entre este modelo e o Random Forest, o segundo melhor modelo. 
# 
# - O modelo Linear Regression foi claramente descartado por apresentar valores muito piores que os outros. 

# #### 1.9.0 Ajuste e melhorias

# - Sobre os ajustes, é impossível definir a quantidade de ajustes ou a melhor forma, pois as possibilidades são quase infinitas. A apresentação se encerrará em apenas 2 ajustes para demonstrar como será feito por mim

# In[46]:


correlacao_feat = pd.DataFrame(modelo_et.feature_importances_, x_train.columns).sort_values(by=0, ascending = False)


# In[47]:


display(correlacao_feat)

plt.figure(figsize = (15,5))
ax = sns.barplot(x = correlacao_feat.index, y = correlacao_feat[0])
ax.tick_params(axis = 'x', rotation = 90)


# In[48]:


backup_data = backup_data.drop('is_business_travel_ready', axis = 1)
for coluna in backup_data:
    if 'bed_type' in coluna:
        backup_data = backup_data.drop(coluna, axis = 1)

#após ajustes, refazer o aprendizado
y = backup_data['price']
x = backup_data.drop('price', axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1)
#treinar denovo
modelo_et.fit(x_train, y_train)
#testar denovo
previsao = modelo_et.predict(x_test)
print(avaliar_modelo('Extra Trees', y_test, previsao))


# ## 2.0 Deploy:

# ### 1.3 criando o arquivo do modelo e da base de dados já tratada. 

#     Deploy Projeto M.L..ipynb

# In[49]:


x['price'] = y
x.to_csv('projetoML.csv')


# In[50]:


import joblib
joblib.dump(modelo_et, 'modelo.joblib')


# 

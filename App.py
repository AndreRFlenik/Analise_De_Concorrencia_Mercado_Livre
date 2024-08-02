# Importação de bibliotecas

import pandas as pd
import requests
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.express as px
import statsmodels.api as sm
import os

from flask import Flask, render_template, request
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

#Função de WebScraping

def search_mercadolivre(product_name):
    url = f"https://lista.mercadolivre.com.br/{product_name}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    products = soup.find_all('li', {'class': 'ui-search-layout__item'})

    datasearch = []

    for product in products:
        name_element = product.find('h2', {'class': 'ui-search-item__title'})
        name = name_element.text.strip() if name_element else 'Nome não disponível'

        price_element = product.find('span', {'class': 'andes-money-amount'})
        if price_element:
            price_text = price_element.text.strip()
            price_text = price_text.replace('R$', '').replace('.', '').replace(',', '.').strip()
            price = float(price_text)
        else:
            price = 0.0

        avaliation_element = product.find('span', {'class': 'ui-search-reviews__rating-number'})
        avaliation = avaliation_element.text.strip() if avaliation_element else '0'
        avaliation = float(avaliation)

        opinion_element = product.find('span', {'class': 'ui-search-reviews__amount'})
        if opinion_element:
            opinion_text = opinion_element.text.strip()
            opinion_match = re.search(r'\((\d+)\)', opinion_text)
            opinion = int(opinion_match.group(1)) if opinion_match else 0
        else:
            opinion = 0
        opinion = float(opinion)

        link_element = product.find('a', {'class': 'ui-search-link'})
        link = link_element['href'] if link_element else 'Link não disponível'

        # Nova parte para extrair o número de vendas
        sales = 0
        if link != 'Link não disponível':

            # Requisição para a página do produto
            product_response = requests.get(link)
            product_soup = BeautifulSoup(product_response.text, 'html.parser')

            # Encontrar o elemento que contém as vendas
            sales_element = product_soup.find('span', {'class': 'ui-pdp-subtitle'})
            if sales_element:
                # Extrair o número de vendas usando expressões regulares
                sales_match = re.search(r'\+(\d+)\s+vendidos', sales_element.text)
                sales = int(sales_match.group(1)) if sales_match else 0
                sales= float(sales)

                 # Verificar categoria Full
            full_category = 0
            full_icon = product_soup.find('svg', {'class': 'ui-pdp-icon ui-pdp-icon--full ui-pdp-color--GREEN'})
            if full_icon:
                full_category = 1

        datasearch.append([name, price, avaliation, opinion, sales, full_category])

    return datasearch

# Insira os dados do Produto (Nome, Preço Estimado, se é Full)
produto_nome = ' '
produto_preco = [0.0]
produto_full =  [0]









app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/submit', methods=['POST'])
def submit():
    produto_nome = request.form['produto_nome']
    produto_preco = request.form['produto_preco']
    produto_full = request.form['produto_full']
    
    if produto_full == 0:
        efull = 'não'
    else:
        efull = ''

    resultado12 = f'O {produto_nome} com o preço {produto_preco}, {efull} é pertencente a categoria Full!'


    
    if not produto_nome or not produto_preco or not produto_full:
        return render_template('Index.html', error_message="Por favor, preencha todos os campos.")

    # Criação do DataFrame com os produtos concorrentes
    df_pesquisa = pd.DataFrame(columns=['produto_nome','produto_preco', 'produto_nota','produto_votantes','produto_vendas', 'produto_full'])

    # Chamar a função para preencher o dataframe
    data = search_mercadolivre(produto_nome)
    df_pesquisa = pd.concat([df_pesquisa, pd.DataFrame(data, columns=['produto_nome','produto_preco', 'produto_nota','produto_votantes','produto_vendas', 'produto_full'])], ignore_index=True)

    # Remover a coluna 'produto_nome' do DataFrame para permanecer apenas as variáveis numéricas
    df_cluster = df_pesquisa.drop(columns=['produto_nome'])

    # Escalar as variáveis
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cluster)

    # Instanciar o modelo KMeans com o número desejado de clusters
    num_clusters = 3  # Três grupos selecionados
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)

    # Ajustar o modelo aos dados escalados
    kmeans_model.fit(df_scaled)

    # Adicionar rótulos de cluster aos dados
    df_cluster['Cluster'] = kmeans_model.labels_

    # Adicionar a coluna 'Cluster' ao DataFrame df_pesquisa
    df_pesquisa = df_pesquisa.merge(df_cluster[['Cluster']], left_index=True, right_index=True)

    # Tratamento das variáveis
    produto_preco = float(produto_preco)  # Converter para float se necessário
    produto_full = float(produto_full)  # Converter para float se necessário

    # Dados do novo produto
    novo_produto = {
        'produto_nome': [produto_nome],
        'produto_preco': [produto_preco],
        'produto_full': [produto_full]
    }

    # Converter para DataFrame
    df_novo_produto = pd.DataFrame(novo_produto)

    # Selecionar as mesmas variáveis que foram usadas para o treinamento
    variaveis = ['produto_preco', 'produto_full']

    # Verificar se as variáveis estão presentes no DataFrame df_pesquisa
    if not all(var in df_pesquisa.columns for var in variaveis):
        raise ValueError("As variáveis não estão presentes no DataFrame df_pesquisa.")

    # Normalizar os dados
    scaler = StandardScaler()
    df_treinamento = df_pesquisa[variaveis]
    df_treinamento_scaled = scaler.fit_transform(df_treinamento)

    # Normalizar o novo produto
    df_novo_produto[variaveis] = df_novo_produto[variaveis].astype(float)  # Converter para tipo float, se necessário
    df_novo_produto_scaled = scaler.transform(df_novo_produto[variaveis])

    # Treinar o modelo K-means com os dados existentes
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df_treinamento_scaled)

    # Prever o cluster do novo produto
    cluster_novo_produto = kmeans.predict(df_novo_produto_scaled)

    cluster_desejado = cluster_novo_produto[0]

    # Apenas os produtos do cluster pesquisado
    if 'Cluster' in df_pesquisa.columns:
        df_produtos_cluster = df_pesquisa[df_pesquisa['Cluster'] == cluster_desejado].copy()
    else:
        print("A coluna 'Cluster' não está presente no DataFrame df_pesquisa.")

    # Calcular estatísticas descritivas
    estatisticas = df_produtos_cluster.describe()

    # Calcular o indicador com base na média
    produto_indicador_media = (float(produto_preco) / float(df_produtos_cluster['produto_preco'].mean())) * 100

    # Condicional para imprimir o resultado
    if produto_indicador_media > 100:
        resultado1 = f"O preço do produto é: {(produto_indicador_media-100):.2f}% maior que a média"
    elif produto_indicador_media < 100:
        resultado1 = f"O preço do produto é: {produto_indicador_media:.2f}% menor que a média"
    else:
        resultado1 = f"O preço do produto é igual à média"

    # Calcular o indicador com base na mediana
    produto_indicador_mediana = (float(produto_preco) / float(df_produtos_cluster['produto_preco'].median())) * 100
    if produto_indicador_mediana > 100:
        resultado2 = f"O preço do produto é: {(produto_indicador_mediana-100):.2f}% maior que a mediana"
    elif produto_indicador_mediana < 100:
        resultado2 = f"O preço do produto é: {produto_indicador_mediana:.2f}% menor que a mediana"
    else:
        resultado2 = f"O preço do produto é igual à mediana"

    # Calcular o indicador com base no mínimo
    produto_indicador_min = (float(produto_preco) / float(df_produtos_cluster['produto_preco'].min())) * 100
    resultado3 = f"O preço do produto é: {(produto_indicador_min-100):.2f}% maior que o menor preço"

    # Calcular o indicador com base no máximo
    produto_indicador_max = (float(df_produtos_cluster['produto_preco'].max()/float(produto_preco))) * 100
    resultado4 = f"O preço do produto é: {produto_indicador_max:.2f}% menor que o maior preço"

    # Calcular os quartis de produto_preco
    quartis_preco = pd.qcut(df_produtos_cluster['produto_preco'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # Adicionar quartis ao DataFrame
    df_produtos_cluster['Quartil_Preco'] = quartis_preco

    # Agrupar por quartil de preço e calcular estatísticas de produto_vendas e produto_nota
    estatisticas_quartil = df_produtos_cluster.groupby('Quartil_Preco').agg({
        'produto_vendas': ['mean', 'median', 'max'],
        'produto_nota': ['mean', 'median', 'max']
    }).reset_index()

    # Valor da variável independente que você quer verificar
    valor_independente = float(produto_preco)  # Certifique-se de que é float

    # Calcular os quartis
    quartis = df_produtos_cluster['produto_preco'].quantile([0.25, 0.5, 0.75])

    # Definir os limites dos quartis
    q1, q2, q3 = quartis.iloc[0], quartis.iloc[1], quartis.iloc[2]

    # Verificar em qual quartil o valor pertence
    if valor_independente <= q1:
        quartil = 'Q1'
    elif valor_independente <= q2:
        quartil = 'Q2'
    elif valor_independente <= q3:
        quartil = 'Q3'
    else:
        quartil = 'Q4'

    # Resultado
    resultado5 = f'O Preço do produto pertence ao quartil {quartil}'

    
    resultado6 = estatisticas_quartil.to_html(index=False)
    
        # Remover valores ausentes
    correlation_matrix = df_produtos_cluster[['produto_preco', 'produto_nota', 'produto_votantes', 'produto_vendas', 'produto_full']].dropna()

    # Calcular a matriz de correlação
    correlations = correlation_matrix.corr()

   

    resultado7 = correlations.to_html(index=False)
    
    X = df_produtos_cluster[['produto_preco', 'produto_nota', 'produto_votantes']]
    y = df_produtos_cluster['produto_vendas']

    # Converter para tipos numéricos se necessário
    X = X.astype(float)
    y = y.astype(float)

    # Adicionar uma constante para o termo de intercepto
    X = sm.add_constant(X)

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar o modelo de regressão linear multivariável usando statsmodels
    model = sm.OLS(y_train, X_train).fit()

    # Fazer previsões para o conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    mse_reg_multipla = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse_reg_multipla}')

    
    produto_nota = df_produtos_cluster['produto_nota'].mean()
    produto_votantes = df_produtos_cluster['produto_votantes'].mean()
        
        # Adicionando a constante ao novo produto
    novo_produto = pd.DataFrame({
            'const': [1],
            'produto_preco': [produto_preco],
            'produto_nota': [produto_nota],
            'produto_votantes': [produto_votantes],
            
        })

        # Ajustando o DataFrame para ter as mesmas colunas que X_train
    novo_produto = novo_produto[['const', 'produto_preco', 'produto_nota', 'produto_votantes']]

    # Garantir que o novo produto tenha as mesmas colunas que o conjunto de treino
    novo_produto = novo_produto[X_train.columns]
    vendas_previstas_reg_multipla = model.predict(novo_produto)

    # Ajustar a formatação do resultado
    resultado8 =f'Vendas previstas para o novo produto é: {vendas_previstas_reg_multipla[0]} com EMQ de {mse_reg_multipla}'
        
    X = df_produtos_cluster[['produto_preco', 'produto_nota', 'produto_votantes']]
    y = df_produtos_cluster['produto_vendas']

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar o modelo de árvore de decisão
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Fazer previsões para o conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliar o modelo com o MSE
    mse_arvore = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse_arvore}')

   

    produto_nota = df_produtos_cluster['produto_nota'].mean()
    produto_votantes = df_produtos_cluster['produto_votantes'].mean()
        
    novo_produto = pd.DataFrame({
            'produto_preco': [produto_preco],
            'produto_nota': [produto_nota],
            'produto_votantes': [produto_votantes],
            
        })

    # Garantir que o novo produto tenha as mesmas colunas que o conjunto de treino
    novo_produto = novo_produto[X_train.columns]
    vendas_previstas_arvore = model.predict(novo_produto)

    # Ajustar a formatação do resultado
    resultado9 = f'Vendas previstas para o novo produto é: {vendas_previstas_arvore[0]} com EMQ de {mse_arvore}'

    X = df_produtos_cluster[['produto_preco', 'produto_nota', 'produto_votantes']]
    y = df_produtos_cluster['produto_vendas']

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializar e treinar o modelo SVR
    model = SVR(kernel='linear')
    model.fit(X_train, y_train)

    # Fazer previsões para o conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    mse_svm = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse_svm}')

    
    
    produto_nota = df_produtos_cluster['produto_nota'].mean()
    produto_votantes = df_produtos_cluster['produto_votantes'].mean()
        
    novo_produto = pd.DataFrame({
            'produto_preco': [produto_preco],
            'produto_nota': [produto_nota],
            'produto_votantes': [produto_votantes],
            
        })

    # Garantir que o novo produto tenha as mesmas colunas que o conjunto de treino
    novo_produto = novo_produto[X_train.columns]
    
    vendas_previstas_svm = model.predict(novo_produto)
    resultado10 = (f'Vendas previstas para o novo produto é: {vendas_previstas_svm[0]} com EMQ de {mse_svm}')

    # Normalização dos EMQs
    max_mse = max(mse_reg_multipla, mse_arvore, mse_svm)
    nemq_reg_multipla = mse_reg_multipla / max_mse
    nemq_arvore = mse_arvore / max_mse
    nemq_svm = mse_svm / max_mse

    # Cálculo do indicador
    indicador = (
        vendas_previstas_reg_multipla * (1 / nemq_reg_multipla) +
        vendas_previstas_arvore * (1 / nemq_arvore) +
        vendas_previstas_svm * (1 / nemq_svm)
    ) / (
        (1 / nemq_reg_multipla) + (1 / nemq_arvore) + (1 / nemq_svm)
    )

    resultado11 = (f"O valor de vendas do indicador combinado é: {float(indicador)}")
    



    return render_template('Index.html', resultado1=resultado1, resultado2=resultado2, resultado3=resultado3, resultado4=resultado4, 
                    resultado5=resultado5, resultado6=resultado6, resultado7=resultado7, resultado8=resultado8, 
                          resultado9=resultado9, resultado10=resultado10, resultado11=resultado11, resultado12=resultado12 )


    





if __name__ == '__main__':
    app.run(debug=True)





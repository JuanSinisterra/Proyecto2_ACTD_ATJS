# Tarea 5 - ACTD

import dash
from dash import dcc  # dash core components
from dash import html  # dash html components
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import shap
import numpy as np
from tensorflow.keras.models import load_model

import psycopg2
import pandas.io.sql as sqlio

engine = psycopg2.connect(
    dbname="",
     user="",
     password="",
     host="",
     port="5432"
)

query = """
SELECT *
FROM tablename;
"""
datos_graficas = sqlio.read_sql_query(query, engine)

for i in range(2,12):
    columna = 'x' + str(i)
    datos_graficas[columna]=pd.to_numeric(datos_graficas[columna])

for i in range(24,30):
    columna = 'x' + str(i)
    datos_graficas[columna]=pd.to_numeric(datos_graficas[columna])

#Hombres y mujeres
datos_hombres = datos_graficas[datos_graficas['x2'] == 1]
y_hombres = datos_hombres['y'].value_counts()
default_hombres = ((y_hombres[1])/(y_hombres[1]+y_hombres[0]))*100

datos_mujeres = datos_graficas[datos_graficas['x2'] == 2]
y_mujeres = datos_mujeres['y'].value_counts()
default_mujeres = ((y_mujeres[1])/(y_mujeres[1]+y_mujeres[0]))*100

dicc_genero = {'Hombres':default_hombres, 'Mujeres': default_mujeres}

#Nivel educativo
datos_postgrado = datos_graficas[datos_graficas['x3'] == 1]
y_postgrado = datos_postgrado['y'].value_counts()
default_postgrado = ((y_postgrado[1])/(y_postgrado[1]+y_postgrado[0]))*100

datos_universitario = datos_graficas[datos_graficas['x3'] == 2]
y_universitario = datos_universitario['y'].value_counts()
default_universitario = ((y_universitario[1])/(y_universitario[1]+y_universitario[0]))*100

datos_bachiller = datos_graficas[datos_graficas['x3'] == 3]
y_bachiller = datos_bachiller['y'].value_counts()
default_bachiller = ((y_bachiller[1])/(y_bachiller[1]+y_bachiller[0]))*100

dicc_educativo = {'Postgrado': default_postgrado, 'Universitario': default_universitario, 'Bachillerato': default_bachiller}

#Estado Marital
datos_casados = datos_graficas[datos_graficas['x4'] == 1]
y_casados = datos_casados['y'].value_counts()
default_casados = ((y_casados[1])/(y_casados[1]+y_casados[0]))*100

datos_solteros = datos_graficas[datos_graficas['x4'] == 2]
y_solteros = datos_solteros['y'].value_counts()
default_solteros = ((y_solteros[1])/(y_solteros[1]+y_solteros[0]))*100

dicc_marital = {'Casados':default_casados, 'Solteros': default_solteros}

#Estado Deuda
diccionario_estado = {}
lista_aldia = []
for i in range(6,12):
    columna = 'x' + str(i)
    datos_it =datos_graficas[datos_graficas[columna] == -1] 
    y_it = datos_it['y'].value_counts()
    if (y_it[1]+y_it[0])>0:
        default_it = ((y_it[1])/(y_it[1]+y_it[0]))*100
        lista_aldia.append(default_it)
default_aldia = sum(lista_aldia)/len(lista_aldia)
diccionario_estado['Al dia'] = default_aldia

lista_1= []
for i in range(6,12):
    columna = 'x' + str(i)
    datos_it =datos_graficas[datos_graficas[columna] == 1] 
    y_it = datos_it['y'].value_counts()
    if len(y_it)!=0:
        default_it = ((y_it[1])/(y_it[1]+y_it[0]))*100
        lista_1.append(default_it)
default_1 = sum(lista_1)/len(lista_1)
diccionario_estado['1 mes atrasado'] = default_1

lista_2 = []
for i in range(6,12):
    columna = 'x' + str(i)
    datos_it =datos_graficas[datos_graficas[columna] == 2] 
    y_it = datos_it['y'].value_counts()
    if (y_it[1]+y_it[0])>0:
        default_it = ((y_it[1])/(y_it[1]+y_it[0]))*100
        lista_2.append(default_it)
default_2 = sum(lista_2)/len(lista_2)
diccionario_estado['2 meses atrasado'] = default_2

lista_3 = []
for i in range(6,12):
    columna = 'x' + str(i)
    datos_it =datos_graficas[datos_graficas[columna] == 3] 
    y_it = datos_it['y'].value_counts()
    if (y_it[1]+y_it[0])>0:
        default_it = ((y_it[1])/(y_it[1]+y_it[0]))*100
        lista_3.append(default_it)
default_3 = sum(lista_3)/len(lista_3)
diccionario_estado['3 meses atrasado'] = default_3


lista_4 = []
for i in range(6,12):
    columna = 'x' + str(i)
    datos_it =datos_graficas[datos_graficas[columna] == 4] 
    y_it = datos_it['y'].value_counts()
    if (y_it[1]+y_it[0])>0:
        default_it = ((y_it[1])/(y_it[1]+y_it[0]))*100
        lista_4.append(default_it)
default_4 = sum(lista_4)/len(lista_4)
diccionario_estado['4 meses atrasado'] = default_4

lista_5 = []
for i in range(6,12):
    columna = 'x' + str(i)
    datos_it =datos_graficas[datos_graficas[columna] == 5] 
    y_it = datos_it['y'].value_counts()
    if (y_it[1]+y_it[0])>0:
        default_it = ((y_it[1])/(y_it[1]+y_it[0]))*100
        lista_5.append(default_it)
default_5 = sum(lista_5)/len(lista_5)
diccionario_estado['5 meses atrasado'] = default_5


lista_6 = []
for i in range(6,12):
    columna = 'x' + str(i)
    datos_it =datos_graficas[datos_graficas[columna] == -1] 
    y_it = datos_it['y'].value_counts()
    if (y_it[1]+y_it[0])>0:
        default_it = ((y_it[1])/(y_it[1]+y_it[0]))*100
        lista_6.append(default_it)
default_6 = sum(lista_6)/len(lista_6)
diccionario_estado['6 meses atrasado'] = default_6

lista_7 = []
for i in range(6,12):
    columna = 'x' + str(i)
    datos_it =datos_graficas[datos_graficas[columna] == -1] 
    y_it = datos_it['y'].value_counts()
    if (y_it[1]+y_it[0])>0:
        default_it = ((y_it[1])/(y_it[1]+y_it[0]))*100
        lista_7.append(default_it)
default_7 = sum(lista_7)/len(lista_7)
diccionario_estado['7 meses atrasado'] = default_7

lista_8 = []
for i in range(6,12):
    columna = 'x' + str(i)
    datos_it =datos_graficas[datos_graficas[columna] == -1] 
    y_it = datos_it['y'].value_counts()
    if (y_it[1]+y_it[0])>0:
        default_it = ((y_it[1])/(y_it[1]+y_it[0]))*100
        lista_8.append(default_it)
default_8 = sum(lista_8)/len(lista_8)
diccionario_estado['8 meses atrasado'] = default_8

# Estado de pago
diccionario_pagos = {}
lista_pago = []
for i in range(24,30):
    columna = 'x' + str(i)
    datos_it =datos_graficas[datos_graficas[columna] == 1] 
    y_it = datos_it['y'].value_counts()
    if (y_it[1]+y_it[0])>0:
        default_it = ((y_it[1])/(y_it[1]+y_it[0]))*100
        lista_pago.append(default_it)
default_pago = sum(lista_pago)/len(lista_pago)
diccionario_pagos['Pago'] = default_pago

lista_no_pago= []
for i in range(24,30):
    columna = 'x' + str(i)
    datos_it =datos_graficas[datos_graficas[columna] == 0] 
    y_it = datos_it['y'].value_counts()
    if (y_it[1]+y_it[0])>0:
        default_it = ((y_it[1])/(y_it[1]+y_it[0]))*100
        lista_no_pago.append(default_it)
default_no_pago = sum(lista_no_pago)/len(lista_no_pago)
diccionario_pagos['No Pago'] = default_no_pago

#Extraer datos maximos
datos_negativos={}

max_genero = max(dicc_genero.items(), key=lambda x: x[1])
llave_genero, valor_genero = max_genero
datos_negativos[llave_genero] = valor_genero

max_educativo = max(dicc_educativo.items(), key=lambda x: x[1])
llave_educ, valor_educ = max_educativo
datos_negativos[llave_educ] = valor_educ

max_marital = max(dicc_marital.items(), key=lambda x: x[1])
llave_marital, valor_marital = max_marital
datos_negativos[llave_marital] = valor_marital

max_estado = max(diccionario_estado.items(), key=lambda x: x[1])
llave_estado, valor_estado = max_estado
datos_negativos[llave_estado] = valor_estado

max_pago = max(diccionario_pagos.items(), key=lambda x: x[1])
llave_pago, valor_pago = max_pago
datos_negativos[llave_pago] = valor_pago

# Sort data by values to identify the two highest values
sorted_data = sorted(datos_negativos.items(), key=lambda x: x[1], reverse=True)

# Extract the two highest values
highest_values = [sorted_data[0][1], sorted_data[1][1]]

# Define color scale
colors = []
for key, value in datos_negativos.items():
    if value in highest_values:
        colors.append('#660000')  # Darkest shade for highest values
    else:
        colors.append('#FFCCCB')

# Create labels for each bar
labels = [f"{value:.2f}" for value in datos_negativos.values()]

# Create the Plotly figure
fig2 = go.Figure(data=[go.Bar(x=list(datos_negativos.keys()), y=list(datos_negativos.values()), text=labels, textposition='auto', marker=dict(color=colors))],
                layout=go.Layout(title='Caracteristicas Negativas'))

#Extraer datos minimos
datos_positivos={}

min_genero = min(dicc_genero.items(), key=lambda x: x[1])
llave_genero, valor_genero = min_genero
datos_positivos[llave_genero] = valor_genero

min_educativo = min(dicc_educativo.items(), key=lambda x: x[1])
llave_educ, valor_educ = min_educativo
datos_positivos[llave_educ] = valor_educ

min_marital = min(dicc_marital.items(), key=lambda x: x[1])
llave_marital, valor_marital = min_marital
datos_positivos[llave_marital] = valor_marital

min_estado = min(diccionario_estado.items(), key=lambda x: x[1])
llave_estado, valor_estado = min_estado
datos_positivos[llave_estado] = valor_estado

min_pago = min(diccionario_pagos.items(), key=lambda x: x[1])
llave_pago, valor_pago = min_pago
datos_positivos[llave_pago] = valor_pago



# Sort data by values to identify the two lowest values
sorted_data = sorted(datos_positivos.items(), key=lambda x: x[1])

# Extract the two lowest values
lowest_values = [sorted_data[0][1], sorted_data[1][1]]

# Define color scale
colors = []
for key, value in datos_positivos.items():
    if value in lowest_values:
        colors.append('#004d00')  # Darker shade for lowest values
    else:
        colors.append('#abf7b1')  # Lighter shade for other values

# Create labels for each bar
labels = [f"{value:.2f}" for value in datos_positivos.values()]

# Create the Plotly figure
fig = go.Figure(data=[go.Bar(x=list(datos_positivos.keys()), y=list(datos_positivos.values()), text=labels, textposition='auto', marker=dict(color=colors))],
                layout=go.Layout(title='Caracteristicas Positivas'))






external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

model = load_model("modelo_entrenado.h5")


df=datos_graficas

X = df[["x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x24","x25","x26","x27","x28","x29"]]
Y = df['y']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Definir el diseño de la aplicación
app.layout = html.Div([
    html.Div([
        html.H3("Producto Toma de Decisiones - Riesgo default en Pago de Creditos",style={'fontWeight': 'bold'}),
        html.H6("Objetivo del Producto", style={'fontWeight': 'bold'}),
        html.H6("El producto busca predecir si un pago hace default dependiendo de caracteristicas del cliente y su historial de pagos .", style={'fontSize': '18px'}),
        html.H6("Instrucciones de uso", style={'fontWeight': 'bold'}),
        html.H6("Seleccione las caracteristicas del cliente a partir de las listas.", style={'fontSize': '18px'}),
        html.H6("Ingrese la edad del cliente en años. [18,120].", style={'fontSize': '18px'}),
    ], style={'margin-bottom': '20px'}),
    html.Hr(),
    html.Div(html.H6("Inicio de la Aplicación", style={'fontWeight': 'bold'})),
    html.Div([
        html.H6("Edad del cliente (años)", style={'fontSize': '18px'}),
        dcc.Input(
            id='X2',
            value=18,
            type='number',
            style={'width': '150px'}
        )
    ], style={'grid-column': '1', 'grid-row': '1'}),
    html.Div([
        html.H6("Género", style={'fontSize': '18px'}),
        dcc.Dropdown(
            options=[
                {'label': 'Masculino', 'value': 1},
                {'label': 'Femenino', 'value': 2}
            ],
            value=1,
            id='X3',
            style={'width': '150px'}
        )
    ],style={'grid-column': '1', 'grid-row': '2'}),
    html.Div([
        html.H6("Educación", style={'fontSize': '18px'}),
        dcc.Dropdown(
            options=[
                {'label': 'Postgrado', 'value': 1},
                {'label': 'Universtario', 'value': 2},
                {'label': 'Bachillerato', 'value': 3},
                {'label': 'Otro', 'value': 4}
            ],
            value=1,
            id='X4',
            style={'width': '150px'}
        )
    ],style={'grid-column': '2', 'grid-row': '1'}),
    html.Div([
        html.H6("Estado Marital", style={'fontSize': '18px'}),
        dcc.Dropdown(
            options=[
                {'label': 'Casado/a', 'value': 1},
                {'label': 'Soltero/a', 'value': 2},
                {'label': 'Otro', 'value': 3}
            ],
            value=1,
            id='X5',
            style={'width': '150px'}
        )
    ],style={'grid-column': '2', 'grid-row': '2'}),
    html.Div([
        html.H6("Estado de la deuda hace 6 meses", style={'fontSize': '18px'}),
        dcc.Dropdown(
            options=[
                {'label': 'Al dia', 'value': -1},
                {'label': '1 mes atrasado', 'value': 1},
                {'label': '2 meses atrasado', 'value': 2},
                {'label': '3 mes atrasado', 'value': 3},
                {'label': '4 meses atrasado', 'value': 4},
                {'label': '5 mes atrasado', 'value': 5},
                {'label': '6 meses atrasado', 'value': 6},
                {'label': '7 mes atrasado', 'value': 7},
                {'label': '8 meses atrasado', 'value': 8},
                {'label': '9 meses o más de atraso', 'value': 9},
            ],
            value=1,
            id='X6',
            style={'width': '200px'}
        )
    ],style={'grid-column': '3', 'grid-row': '1'}),
    html.Div([
        html.H6("Estado de la deuda hace 5 meses", style={'fontSize': '18px'}),
        dcc.Dropdown(
            options=[
                {'label': 'Al dia', 'value': -1},
                {'label': '1 mes atrasado', 'value': 1},
                {'label': '2 meses atrasado', 'value': 2},
                {'label': '3 mes atrasado', 'value': 3},
                {'label': '4 meses atrasado', 'value': 4},
                {'label': '5 meses atrasado', 'value': 5},
                {'label': '6 meses atrasado', 'value': 6},
                {'label': '7 meses atrasado', 'value': 7},
                {'label': '8 meses atrasado', 'value': 8},
                {'label': '9 meses o más de atraso', 'value': 9},
            ],
            value=1,
            id='X7',
            style={'width': '200px'}
        )
    ],style={'grid-column': '3', 'grid-row': '2'}),
    html.Div([
        html.H6("Estado de la deuda hace 4 meses", style={'fontSize': '18px'}),
        dcc.Dropdown(
            options=[
                {'label': 'Al dia', 'value': -1},
                {'label': '1 mes atrasado', 'value': 1},
                {'label': '2 meses atrasado', 'value': 2},
                {'label': '3 meses atrasado', 'value': 3},
                {'label': '4 meses atrasado', 'value': 4},
                {'label': '5 meses atrasado', 'value': 5},
                {'label': '6 meses atrasado', 'value': 6},
                {'label': '7 meses atrasado', 'value': 7},
                {'label': '8 meses atrasado', 'value': 8},
                {'label': '9 meses o más de atraso', 'value': 9},
            ],
            value=1,
            id='X8',
            style={'width': '200px'}
        )
    ],style={'grid-column': '4', 'grid-row': '1'}),
    html.Div([
        html.H6("Estado de la deuda hace 3 meses", style={'fontSize': '18px'}),
        dcc.Dropdown(
            options=[
                {'label': 'Al dia', 'value': -1},
                {'label': '1 mes atrasado', 'value': 1},
                {'label': '2 meses atrasado', 'value': 2},
                {'label': '3 meses atrasado', 'value': 3},
                {'label': '4 meses atrasado', 'value': 4},
                {'label': '5 meses atrasado', 'value': 5},
                {'label': '6 meses atrasado', 'value': 6},
                {'label': '7 meses atrasado', 'value': 7},
                {'label': '8 meses atrasado', 'value': 8},
                {'label': '9 meses o más de atraso', 'value': 9},
            ],
            value=1,
            id='X9',
            style={'width': '200px'}
        )
    ],style={'grid-column': '4', 'grid-row': '2'}),
    html.Div([
        html.H6("Estado de la deuda hace 2 meses", style={'fontSize': '18px'}),
        dcc.Dropdown(
            options=[
                {'label': 'Al dia', 'value': -1},
                {'label': '1 mes atrasado', 'value': 1},
                {'label': '2 meses atrasado', 'value': 2},
                {'label': '3 meses atrasado', 'value': 3},
                {'label': '4 meses atrasado', 'value': 4},
                {'label': '5 meses atrasado', 'value': 5},
                {'label': '6 meses atrasado', 'value': 6},
                {'label': '7 meses atrasado', 'value': 7},
                {'label': '8 meses atrasado', 'value': 8},
                {'label': '9 meses o más de atraso', 'value': 9},
            ],
            value=1,
            id='X10',
            style={'width': '200px'}
        )
    ],style={'grid-column': '5', 'grid-row': '1'}),
    html.Div([
        html.H6("Estado de la deuda el mes pasado", style={'fontSize': '18px'}),
        dcc.Dropdown(
            options=[
                {'label': 'Al dia', 'value': -1},
                {'label': '1 mes atrasado', 'value': 1},
                {'label': '2 meses atrasado', 'value': 2},
                {'label': '3 meses atrasado', 'value': 3},
                {'label': '4 meses atrasado', 'value': 4},
                {'label': '5 meses atrasado', 'value': 5},
                {'label': '6 meses atrasado', 'value': 6},
                {'label': '7 meses atrasado', 'value': 7},
                {'label': '8 meses atrasado', 'value': 8},
                {'label': '9 meses o más de atraso', 'value': 9},
            ],
            value=1,
            id='X11',
            style={'width': '200px'}
        )
    ],style={'grid-column': '5', 'grid-row': '2'}),
    html.Div([
        html.H6("Se abonó a la deuda hace 6 meses?", style={'fontSize': '18px'}),
        dcc.Dropdown(
            options=[
                {'label': 'Si', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=1,
            id='X24',
            style={'width': '150px'}
        )
    ],style={'grid-column': '6', 'grid-row': '1'}),
    html.Div([
        html.H6("Se abonó a la deuda hace 5 meses?", style={'fontSize': '18px'}),
        dcc.Dropdown(
            options=[
                {'label': 'Si', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=1,
            id='X25',
            style={'width': '150px'}
        )
    ],style={'grid-column': '6', 'grid-row': '2'}),
    html.Div([
        html.H6("Se abonó a la deuda hace 4 meses?", style={'fontSize': '18px'}),
        dcc.Dropdown(
            options=[
                {'label': 'Si', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=1,
            id='X26',
            style={'width': '150px'}
        )
    ],style={'grid-column': '7', 'grid-row': '1'}),
    html.Div([
        html.H6("Se abonó a la deuda hace 3 meses?", style={'fontSize': '18px'}),
        dcc.Dropdown(
            options=[
                {'label': 'Si', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=1,
            id='X27',
            style={'width': '150px'}
        )
    ],style={'grid-column': '7', 'grid-row': '2'}),
    html.Div([
        html.H6("Se abonó a la deuda hace 2 meses?", style={'fontSize': '18px'}),
        dcc.Dropdown(
            options=[
                {'label': 'Si', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=1,
            id='X28',
            style={'width': '150px'}
        )
    ],style={'grid-column': '8', 'grid-row': '1'}),
    html.Div([
        html.H6("Se abonó a la deuda el mes pasado?", style={'fontSize': '18px'}),
        dcc.Dropdown(
            options=[
                {'label': 'Si', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=1,
            id='X29',
            style={'width': '150px'}
        )
    ]),html.Hr(),
    html.Div(html.Div(html.H6("Resultados Preliminares de la Aplicación", style={'fontWeight': 'bold'}))),
    html.Div([
        dcc.Graph(id='grafico_negativo', figure=fig)  
    ]), 
    html.Div([
        dcc.Graph(id='grafico_negativo', figure=fig2)  
    ]),    
    html.Div(html.Br()),

    html.Hr(),
    html.Div(html.Div(html.H6("Resultados de la Aplicación", style={'fontWeight': 'bold'}))),
    html.Div([
        dcc.Graph(id='shap-graph')  
    ]),
    html.Div(id='output-container',style={'fontSize': '20px'}),
       
])

# Definir la función de salida para el gráfico SHAP dinámico
@app.callback(
    Output('shap-graph', 'figure'),
    [Input('X2', 'value'),
     Input('X3', 'value'),
     Input('X4', 'value'),
     Input('X5', 'value'),
     Input('X6', 'value'),
     Input('X7', 'value'),
     Input('X8', 'value'),
     Input('X9', 'value'),
     Input('X10', 'value'),
     Input('X11', 'value'),
     Input('X24', 'value'),
     Input('X25', 'value'),
     Input('X26', 'value'),
     Input('X27', 'value'),
     Input('X28', 'value'),
     Input('X29', 'value')]
)

def update_shap_graph(X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X24, X25, X26, X27, X28, X29):
    # Crear un DataFrame con los valores de entrada proporcionados
    input_data = pd.DataFrame({
    "Género": [X2],
    "Educación": [X3],
    "Estado Marital": [X4],
    "Edad del cliente (años)": [X5],
    "Estado de la deuda hace 6 meses": [X6],
    "Estado de la deuda hace 5 meses": [X7],
    "Estado de la deuda hace 4 meses": [X8],
    "Estado de la deuda hace 3 meses": [X9],
    "Estado de la deuda hace 2 meses": [X10],
    "Estado de la deuda el mes pasado": [X11],
    "Se abonó a la deuda hace 6 meses": [X24],
    "Se abonó a la deuda hace 5 meses": [X25],
    "Se abonó a la deuda hace 4 meses": [X26],
    "Se abonó a la deuda hace 3 meses": [X27],
    "Se abonó a la deuda hace 2 meses": [X28],
    "Se abonó a la deuda el mes pasado": [X29]
})

    explainer = shap.Explainer(model, X_train)
    shap_values_single = explainer.shap_values(input_data)


    # Calcular la importancia media de las características absolutas
    mean_abs_shap_values = np.mean(np.abs(shap_values_single), axis=0)
    
    feature_names = input_data.columns
    
    # Crear un DataFrame de importancia de características
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': mean_abs_shap_values})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Crear la figura de barras para mostrar la importancia de las características
    fig_prelim = go.Figure()
    fig_prelim.add_trace(go.Bar(x=importance_df['Feature'], y=importance_df['Importance'], marker_color='skyblue'))
    fig_prelim.update_layout(title='Importancia de características - Modelo Predictivo', xaxis_title='Característica', yaxis_title='Importancia', height=600, width=1200)

    return fig_prelim

@app.callback(
    [Output('output-container', 'children')],
    [Input('X2', 'value'),
     Input('X3', 'value'),
     Input('X4', 'value'),
     Input('X5', 'value'),
     Input('X6', 'value'),
     Input('X7', 'value'),
     Input('X8', 'value'),
     Input('X9', 'value'),
     Input('X10', 'value'),
     Input('X11', 'value'),
     Input('X24', 'value'),
     Input('X25', 'value'),
     Input('X26', 'value'),
     Input('X27', 'value'),
     Input('X28', 'value'),
     Input('X29', 'value')]
)

def update_output(X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X24, X25, X26, X27, X28, X29):
    input_data = pd.DataFrame({
    "Género": [X2],
    "Educación": [X3],
    "Estado Marital": [X4],
    "Edad del cliente (años)": [X5],
    "Estado de la deuda hace 6 meses": [X6],
    "Estado de la deuda hace 5 meses": [X7],
    "Estado de la deuda hace 4 meses": [X8],
    "Estado de la deuda hace 3 meses": [X9],
    "Estado de la deuda hace 2 meses": [X10],
    "Estado de la deuda el mes pasado": [X11],
    "Se abonó a la deuda hace 6 meses": [X24],
    "Se abonó a la deuda hace 5 meses": [X25],
    "Se abonó a la deuda hace 4 meses": [X26],
    "Se abonó a la deuda hace 3 meses": [X27],
    "Se abonó a la deuda hace 2 meses": [X28],
    "Se abonó a la deuda el mes pasado": [X29]
})
    
    predictions = model.predict(input_data)

    binary_predictions = (predictions > 0.5).astype("int32")

    if binary_predictions[0] == 1:
        prediction_text = "A partir de las características seleccionadas, el modelo predictivo basado en redes neuronales pronostica que cliente HACE default en su tarjeta de crédito."
    else:
        prediction_text = "A partir de las características seleccionadas, el modelo predictivo basado en redes neuronales pronostica que cliente NO HACE default en su tarjeta de crédito."
    
    return [html.Div(prediction_text)]

if __name__ == '__main__':
    app.run_server(debug=True)

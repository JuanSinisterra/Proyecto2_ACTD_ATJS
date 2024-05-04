# Tarea 5 - ACTD

import dash
from dash import dcc  # dash core components
from dash import html  # dash html components
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    html.Div([
        html.H3("Producto Toma de Decisiones - Riesgo default en Pago de Creditos"),
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
            value=0,
            type='number',
            style={'width': '150px'}
        )
    ]),
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
    ]),
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
    ]),
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
    ]),
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
    ]),
                html.Div([
        html.H6("Estado de la deuda hace 5 meses", style={'fontSize': '18px'}),
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
            id='X7',
            style={'width': '200px'}
        )
    ]),
                html.Div([
        html.H6("Estado de la deuda hace 4 meses", style={'fontSize': '18px'}),
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
            id='X8',
            style={'width': '200px'}
        )
    ]),
                html.Div([
        html.H6("Estado de la deuda hace 3 meses", style={'fontSize': '18px'}),
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
            id='X9',
            style={'width': '200px'}
        )
    ]),
                html.Div([
        html.H6("Estado de la deuda hace 2 meses", style={'fontSize': '18px'}),
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
            id='X10',
            style={'width': '200px'}
        )
    ]),
                html.Div([
        html.H6("Estado de la deuda el mes pasado", style={'fontSize': '18px'}),
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
            id='X11',
            style={'width': '200px'}
        )
    ]),
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
    ]),
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
    ]),
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
    ]),
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
    ]),
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
    ]),
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
    ]),    
    html.Hr(),
    html.Div(html.Div(html.H6("Resultados Preliminares la Aplicación", style={'fontWeight': 'bold'}))),
    html.Div(html.Br()),
    html.Div(id='output-container99',style={'fontSize': '20px'}),
    html.Hr(),
    html.Div(html.Div(html.H6("Resultados de la Aplicación", style={'fontWeight': 'bold'}))),
    html.Div(id='output-container',style={'fontSize': '20px'}),
    html.Div(id='output-container2',style={'fontSize': '20px'})
])

import plotly.graph_objs as go
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import sys
from packaging import version
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt

X = df[["X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X24","X25","X26","X27","X28","X29"]]
Y = df['Y']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = keras.Sequential([
    layers.Dense(64, activation='elu', input_shape=(X_train.shape[1],)),
    layers.Dense(8, activation='tanh'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, Y_test)




df=pd.read_csv("Datos Modelamiento.csv")

@app.callback(
    Output('output-container99', 'children'),
    [
        Input('incentivo', 'value'),
        Input('tiempo', 'value'),
        Input('equipo', 'value'),
        Input('departamento', 'value')
    ]
)
def update_output(incentivo, tiempo, equipo, departamento):
    betas_real = [0.794920, -0.185258, -0.018902, -0.003449, 0.004133, -0.032679, 
         -0.062889, -0.054918, -0.051835, -0.051450, -0.045606, -0.061901, 
         -0.021129, -0.000005, 0.004122]
    
    betas_target = [0.731224, -0.089341, 0.012373, 0.014434, -0.002691, -0.022166, 
               0.025077, 0.016419, 0.006372, 0.038684, 0.024695, 0.002289, 
               0.045522, -0.000003, 0.001895]
    
    if tiempo == "":
        tiempo = 0
    
    coeficientes = [1, departamento, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, tiempo, incentivo]
    
    for i in range(1, 14):
        if equipo == i and equipo != 1:
            coeficientes[i] = 1
    
    vector_efectos_real = [betas_real[i] * coeficientes[i] for i in range(15)]
    
    vector_efectos_target = [betas_target[i] * coeficientes[i] for i in range(15)]
    
    x_data = ['Intercepto', 'Departamento', 'Equipo 2', 'Equipo 3', 'Equipo 4',
              'Equipo 5', 'Equipo 6', 'Equipo 7', 'Equipo 8', 'Equipo 9',
              'Equipo 10', 'Equipo 11', 'Equipo 12', 'Tiempo extra', 'Incentivo']
    
    bar_plot_real = go.Bar(x=x_data, y=vector_efectos_real, name='Productividad Real', marker=dict(color='blue'))
    
    bar_plot_target = go.Bar(x=x_data, y=vector_efectos_target, name='Productividad Objetivo', marker=dict(color='red'))
    
    layout = go.Layout(title='Efectos de los parámetros en la productividad', xaxis=dict(title='Variables'), yaxis=dict(title='Efectos'))
    fig = go.Figure(data=[bar_plot_real, bar_plot_target], layout=layout)
    
    return dcc.Graph(figure=fig)


@app.callback(
    Output('output-container', 'children'),
    [
        Input('incentivo', 'value'),
        Input('tiempo', 'value'),
        Input('equipo', 'value'),
        Input('departamento', 'value')
    ]
)
def update_output(incentivo, tiempo, equipo, departamento):
    betas = [0.794920, -0.185258, -0.018902, -0.003449, 0.004133, -0.032679, 
         -0.062889, -0.054918, -0.051835, -0.051450, -0.045606, -0.061901, 
         -0.021129, -0.000005, 0.004122]
    if tiempo=="":
        tiempo=0
    coeficientes=[1,departamento,0,0,0,0,0,0,0,0,0,0,0,tiempo,incentivo]
    for i in range(1,13):
        if equipo==i and equipo!=1:
            coeficientes[i+1]=1
            
    productividad_real=sum(betas * coeficientes for betas, coeficientes in zip(betas, coeficientes))
    
    respuesta3 = html.P("La productividad real para los parámetros dados es: " + str(min(round(productividad_real,2),1)),style={'color': 'blue'})

    return respuesta3

@app.callback(
    Output('output-container2', 'children'),
    [
        Input('incentivo', 'value'),
        Input('tiempo', 'value'),
        Input('equipo', 'value'),
        Input('departamento', 'value')
    ]
)
def update_output2(incentivo, tiempo, equipo, departamento):        
    betas_target=[0.731224, -0.089341, 0.012373, 0.014434, -0.002691, -0.022166, 
               0.025077, 0.016419, 0.006372, 0.038684, 0.024695, 0.002289, 
               0.045522, -0.000003, 0.001895]
    if tiempo=="":
        tiempo=0
    coeficientes_target=[1,departamento,0,0,0,0,0,0,0,0,0,0,0,tiempo,incentivo]
    for i in range(1,13):
        if equipo==i and equipo!=1:
            coeficientes_target[i+1]=1
            
    productividad_target=sum(betas_target * coeficientes_target for betas_target, coeficientes_target in zip(betas_target, coeficientes_target))
    
    respuesta4 = html.P("La productividad objetivo o meta para los parámetros dados es: " + str(min(round(productividad_target,2),1)), style={'color': 'red'})

    return respuesta4

if __name__ == '__main__':
    app.run_server(debug=True)

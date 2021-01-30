# Run this app with `python app.py` 

import numpy as np
import pandas as pd 
#import plotly.express as px
#from jupyter_dash import JupyterDash
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import _io as io
import base64
from sklearn.metrics import confusion_matrix
import pickle
#from shap import force_plot

with open('colnum.pkl', 'rb') as f:
    colnum = pickle.load(f)  

with open('df_client_dash.pkl','rb') as fp:
    df_client= pickle.load(fp)


with open('encoded_logo.pkl','rb') as fp:
    encoded_logo= pickle.load(fp)   
    
with open('encoded_sum.pkl','rb') as fp:
    encoded_sum= pickle.load(fp)   
    
with open('lgbm.pkl','rb') as fp:
    lgbm= pickle.load(fp)   

#definition de liste de variables pour les affichage
colpie=['CODE_GENDER','FLAG_OWN_CAR',
     'FLAG_OWN_REALTY','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE']
coldef = colpie + colnum
coldef.sort()

#couleur gris logo
color = '#D8D8D8'
#couleur bleue
bleue= '#1e88e5'
#couleur rouge
rouge = '#ff0d57'

external_stylesheets = ['https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div(style={'backgroundColor': "color"},children=[
    

                    html.Img(src='data:image/png;base64,{}'.format(encoded_logo)),
                    html.H1(children='Tableau de bord "Prêt à depenser" '),
                    html.Label(['Choisissez un client:'],style={'font-weight': 'bold', "text-align": "left"}),
    
                    dcc.Dropdown(id='dropdown_client', options=[
                    {'label': i, 'value': i} for i in df_client.SK_ID_CURR.sort_values().unique()
                        ], multi=False, placeholder='Please select...',style=dict(width='50%')),
                    dcc.Graph(id='tableau'
                    ),
                    

])

# Tableau de données client
@app.callback(Output('tableau', 'figure'), 
              Input('dropdown_client', 'value'))
def update_table(id_client):
    pd.options.display.float_format = '{:.2f}%'.format
    
    if id_client is not None:

        fig = go.Figure(data=[go.Table(
        header=dict(values= [
                         "<b>GENRE</b>",
                          "<b>SITUATION</b>",
                         "<b>AGE</b>",
                         "<b>ENFANTS</b>", 
                         "<b>REVENU TOTAL</b>", 
                         "<b>TYPE DE REVENU</b>",
                         "<b>ANCIENNETE</b>",
                            ],
                    fill_color='black', line_color='white',
                    align='center',font=dict(color='white', size=10),height=40),
        cells=dict(values = [
                       
                       df_client[df_client.SK_ID_CURR == id_client].CODE_GENDER,
                       df_client[df_client.SK_ID_CURR == id_client].NAME_FAMILY_STATUS,
                       df_client[df_client.SK_ID_CURR == id_client].YEARS_BIRTH.values.astype(int),
                       df_client[df_client.SK_ID_CURR == id_client].CNT_CHILDREN, 
                       df_client[df_client.SK_ID_CURR == id_client].AMT_INCOME_TOTAL, 
                       df_client[df_client.SK_ID_CURR == id_client].NAME_INCOME_TYPE, 
                       df_client[df_client.SK_ID_CURR == id_client].YEARS_EMPLOYED.apply('{:.1f}'.format),
                            ],
                   fill_color=color,line_color='white',
                   font=dict(color='black', size=12),height=40,align=['center'],))
        ])
        fig.update_layout(height = 100,margin=dict(l=0, r=20, t=0, b=0))

        return fig
    else:
        fig = go.Figure(data=[go.Table(
        header=dict(values= [
                         "<b>GENRE</b>",
                          "<b>SITUATION</b>",
                         "<b>AGE</b>",
                         "<b>ENFANTS</b>", 
                         "<b>REVENU TOTAL</b>", 
                         "<b>TYPE DE REVENU</b>",
                         "<b>ANCIENNETE</b>"],
                    fill_color='black', line_color='white',
                    align='center',font=dict(color='white', size=10),height=40),
        cells=dict(values=['','','','','','',''],
                   fill_color=color,line_color='white',
                   font=dict(color='white', size=12),height=40,align=['center'],))
        ])
        fig.update_layout(height = 100,margin=dict(l=0, r=0, t=0, b=0))


        return fig
        
#app.run_server(mode='external',debug=True, use_reloader=False)        
#app.run_server(mode='inline', port=8069)

if __name__ == '__main__':
    app.run_server(debug=True)
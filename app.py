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
from shap import force_plot


with open('df_client_dash.pkl','rb') as fp:
    df_client= pickle.load(fp)



with open('encoded_logo.pkl','rb') as fp:
    encoded_logo= pickle.load(fp)   

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
                    

])


        
#app.run_server(mode='external',debug=True, use_reloader=False)        
#app.run_server(mode='inline', port=8069)

if __name__ == '__main__':
    app.run_server(debug=True)
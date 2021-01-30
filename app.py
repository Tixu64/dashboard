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

with open('select_features.pkl','rb') as fp:
    selected_features = pickle.load(fp)

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

with open('encoded_sum.pkl','rb') as fp:
    encoded_sum= pickle.load(fp)

def display_summary_plot(encoded_sum) : 
    #plt.close('all') #pour éviter les superposition de graphiques
    shap_html =''
    #summary_plot = shap.summary_plot(shap_values[1], df_client[selected_features],plot_type="violin", 
                                     #show=False,max_display=10)
    #color='RdGy',
    """ figure to html base64 png image """ 
    #tmpfile_sum = io.BytesIO()
    #plt.tight_layout()
    #plt.savefig(tmpfile_sum, format='png',facecolor='w',transparent=False,edgecolor=color,dpi=70)
    #encoded_sum = base64.b64encode(tmpfile_sum.getvalue()).decode('utf-8')
    shap_html = html.Img(src=f"data:image/png;base64, {encoded_sum}")
    #del tmpfile_sum
    #del summary_plot
    #del encoded_sum
    #plt.close('all')
    return shap_html


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
                    dcc.Markdown(id='display_loan',style={'marginLeft': 50 }),
                    html.H5(children='Probabilité de défaut de crédit',style={'color': rouge }),
                    
                    dcc.Graph(id='indicateur',style={'text-align':'center'}),
                                        dcc.Markdown('''  
**Seuil de défaut : 49.4 %**  
**Seuil au minimum de pertes attendues : 73%**  
*Taux de perte si défaut de crédit: 70%*      
*Taux de perte si refus de crédit : 20%* ''',style={'marginLeft': 50,'color': 'gray','margin-top':'10px','font-size': '12px'} ),

                    dcc.Markdown(id='disp_prob'),
                    html.Label(children='''
                       Variables importantes du modèle
                   ''',style={'marginLeft': 50,'text-align':'left'}),
                   html.Label(children='''
                           ROUGE: valeurs hautes de la variable                                                        
                                   ''',style={'marginLeft': 50,'text-align':'left','color': rouge,'font-size': '12px'}),
                 html.Label(children='''
                           BLEU: valeurs basses de la variable                                                       
                                   ''',style={'marginLeft': 50,'text-align':'left','color': bleue,'font-size': '12px'}),
                        #affichage du summer plot shap violon
 
                html.Div(children = display_summary_plot(encoded_sum),style={"border":"2px"}),

])
                                                     
#indicateur de probalilité
@app.callback(Output('indicateur', 'figure'), 
              Input('dropdown_client', 'value'))
def indicateur(selected_value):
    
    if selected_value is not None:
            proba = lgbm.predict(df_client[selected_features][df_client.SK_ID_CURR == selected_value])[0]
    else:
            proba = 0
    
    fig = go.Figure(go.Indicator(
    mode = "number+gauge",
    gauge= {'shape': "bullet",
             'axis': {'range': [None, 100]},
    
             'bar': {'color': rouge,'thickness': 0.60},
             'steps' : [{'range': [49.4,100], 'color': color}],
             'threshold': {
                'line': {'color': rouge, 'width': 2},
                'thickness': 0.8, 'value':73},
                         },
    delta = {'reference': 150},
    value = proba*100,
    domain = {'x': [0.1, 1], 'y': [0, 1]},
    title = {'text': "%"}))
    fig.update_layout(height = 70,margin=dict(l=20, r=0, t=0, b=20))
    return fig

#call back affichage des fp et fn
@app.callback(Output('display_loan', 'children'), 
              Input('dropdown_client', 'value'))
def display_loan(selected_value):
         
        if selected_value is not None:
    
            loan_val = df_client['AMT_CREDIT'][df_client.SK_ID_CURR == selected_value].values.astype(int)[0]
            loan_price = df_client['AMT_GOODS_PRICE'][df_client.SK_ID_CURR == selected_value].values.astype(int)[0]
            loan_annuity = df_client['AMT_ANNUITY'][df_client.SK_ID_CURR == selected_value].values.astype(int)[0]
            loan_lenght = df_client['LENGTH_CREDIT'][df_client.SK_ID_CURR == selected_value].values.astype(int)[0]
            loan_cnt = df_client['CNT_ACTIVE_LOAN'][df_client.SK_ID_CURR == selected_value].values.astype(int)[0]
    
           

            return ('''
> ###### Montant emprunté   : __{:,.0f}__  
> - Mensualité        : {:,.0f}      
> - Montant à financer: {:,.0f}   
> - Durée             : {:,.1f}  
> - Nombre de crédit  : {:,.0f}  
      
'''.format(loan_val,loan_annuity,loan_price,loan_lenght,loan_cnt))
        else:
            loan_val =0
            
            return ('''    
   
  ''')    



#call back affichage des fp et fn
@app.callback(Output('disp_prob', 'children'), 
              Input('dropdown_client', 'value'))
def display_proba(selected_value):
    
        loss_given_def = 0.7 #arbitraire à fixer avec le metier
        gain_no_def = 0.2 #arbitraire à fixer avec le metier
        threshold = 0.735 #proba de perte minimale calculee sur le jeu de test        
        if selected_value is not None:
    
            loan_val = df_client['AMT_CREDIT'][df_client.SK_ID_CURR == selected_value].values.astype(int)[0]
            
            proba = lgbm.predict(df_client[selected_features][df_client.SK_ID_CURR == selected_value])[0]
            y = df_client.TARGET
    
            y_pred_prob = lgbm.predict(df_client[selected_features])
    
            opp = 0
            real = 0
            total = 0
        
            
            #calcul pour la probabilité de defaut du client
            prediction = y_pred_prob > proba

            cm = confusion_matrix(y, prediction)
            
            pcm = cm/np.sum(cm)
            
            pfp = pcm[1,0]
            pfn = pcm[0,1]

            return ('''     
> ###### Pour un seuil de défaut de __{:,.1f}%__  
> - Mauvaise prédiction de solvabilité : {:,.0f}%  
> - Mauvaise prédiction de défaut: {:,.0f}%  
'''.format(proba*100,pfp*100,pfn*100))
        else:
            loan_val =0
            pfp = 0
            pfn = 0
            proba = 0
            
            return (''' 
   
  ''')

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
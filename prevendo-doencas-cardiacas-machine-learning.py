# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:42:00 2024

@author: leoja
"""

#%% Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%% Leitura dos dados
df_cardio = pd.read_csv(r'C:\Users\leoja\OneDrive\Documentos\Asimov Academy\Aulas\Trilha Data Science & Machine Learning\11 - Prevendo risco de doenças cardíacas com Machine Learning\Análise de Doenças Cardíacas com Machine Learning\cardio_train.csv',index_col=0,sep=',')

# Análise dos dados
df_cardio.info()
df_cardio.describe()
df_cardio.isna().sum()

#%% Análise exploratória
# Dados numéricos
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

fig = make_subplots(rows=4, cols=1)
fig.add_trace(go.Box(x=df_cardio['age']/365, name='Idade'),row=1, col=1)
fig.add_trace(go.Box(x=df_cardio['weight'], name='Peso'),row=2, col=1)
fig.add_trace(go.Box(x=df_cardio['ap_hi'], name='Pressão sanguínea sistólica'),row=3, col=1)
fig.add_trace(go.Box(x=df_cardio['ap_lo'], name='Pressão sanguínea diastólica'),row=4, col=1)

fig.update_layout(height=700)
fig.show()

# Dados categóricos
fig = make_subplots(rows=2, cols=3)
fig.add_trace(go.Bar(y=df_cardio['gender'].value_counts(), x=['Feminino','Masculino'], name='Genero'),row=1, col=1)
fig.add_trace(go.Bar(y=df_cardio["cholesterol"].value_counts(), x=["Normal", "Acima do Normal", "Muito acima do normal"], name="Cholesterol"), row=1, col=2)
fig.add_trace(go.Bar(y=df_cardio["gluc"].value_counts(), x=["Normal", "Acima do Normal", "Muito acima do normal"], name="Glicose"), row=1, col=3)
fig.add_trace(go.Bar(y=df_cardio["smoke"].value_counts(), x=["Não fumante", "Fumante"], name="Fumante"), row=2, col=1)
fig.add_trace(go.Bar(y=df_cardio["alco"].value_counts(), x=["Não Alcoólatra", "Alcoólatra"], name="Alcoólatra"), row=2, col=2)
fig.add_trace(go.Bar(y=df_cardio["active"].value_counts(), x=["Não Ativo", "Ativo"], name="Ativo"), row=2, col=3)

fig.update_layout(height=700)
fig.show()

# Variável de estudo
df_cardio['cardio'].value_counts()
df_cardio.groupby(['smoke','cardio']).count()['id']


#%% Machine Learning
Y = df_cardio['cardio']
cols_sel = [i for i in df_cardio.columns if i != 'cardio']
X = df_cardio[cols_sel]

# Separa os dados em dados de treino e de teste
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Treinamento do modelo
from sklearn.ensemble import RandomForestClassifier

ml_model = RandomForestClassifier(n_estimators=20, n_jobs=4, max_depth=4)
ml_model.fit(x_train, y_train)

x_test.iloc[0].to_frame().transpose()
ml_model.predict(x_test.iloc[0].to_frame().transpose())
y_test.iloc[0]

# Avaliação do modelo
from sklearn.metrics import classification_report, confusion_matrix

predictions = ml_model.predict(x_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Feature importance
from sklearn.inspection import permutation_importance

result = permutation_importance(ml_model, x_test, y_test, n_repeats=10, n_jobs=2)
sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=x_test.columns[sorted_idx])
ax.set_title('Permutation Importances (test set)')
fig.tight_layout()
plt.show()

# Gráfico que mostra qual variável tem mais peso no resultado
import shap
explainer = shap.TreeExplainer(ml_model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values[:, :, 1], X)









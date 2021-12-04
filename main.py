from matplotlib.pyplot import switch_backend
import streamlit as st

import Asociacion
import Distancia
import Clustering

st.title("Inteligencia Artificial")

# #Icono a la pestana y efectos de pestana
# st.set_page_config(page_icon=":brain:", page_title="IntelliShow", initial_sidebar_state="collapsed",)


st.sidebar.title('Módulos Inteligencia Artificial')
modulo=st.sidebar.selectbox('Selecciona el módulo con el que deseas trabajar:',('Reglas de asociación','Métricas de distancia', 'Clustering', 'Clasificación', 'Árboles de Decisión'))
#physics=st.sidebar.checkbox('add physics interactivity?')
#got.simple_func(physics)

if modulo=='Reglas de asociación':
  Asociacion.programa()

if modulo=='Métricas de distancia':
    Distancia.programa()

if modulo=='Clustering':
    Clustering.programa()

#if modulo=='Clasificación':

if modulo=='Árboles de Decisión':
    tipo_arb=st.sidebar.selectbox('Selecciona el algoritmo que deseas trabajar:',('Pronóstico','Clasificación'))
    # if tipo_arb=='Pronóstico':
    # if tipo_arb=='Clasificación':

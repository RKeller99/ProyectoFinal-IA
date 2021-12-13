from matplotlib.pyplot import switch_backend
import streamlit as st

import Asociacion
import Distancia
import Clustering
import Regresion
import Arboles

st.set_page_config(page_title="Inteligencia Artificial - RK")
st.title("Inteligencia Artificial")
st.sidebar.title('Módulos Inteligencia Artificial')
modulo=st.sidebar.selectbox('Selecciona el módulo con el que deseas trabajar:',('Reglas de asociación','Métricas de distancia', 'Clustering', 'Regresión Logística', 'Árboles de Decisión'))

if modulo=='Reglas de asociación':
  Asociacion.programa()

if modulo=='Métricas de distancia':
    Distancia.programa()

if modulo=='Clustering':
    Clustering.programa()

if modulo=='Regresión Logística':
    Regresion.programa()

if modulo=='Árboles de Decisión':
    Arboles.programa()

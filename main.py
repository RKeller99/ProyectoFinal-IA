import streamlit as st

import Asociacion
import Distancia
import Clustering

st.title("Inteligencia Artificial")

st.sidebar.title('Menú de Opciones')

#TABS = {"Inicio": Bienvenida, "Métricas Distancia": MetricasDistancia, "Reglas de Asociación (A priori)": ReglasAsociacion, "Clústering Jerárquico": ClusteringJerarquico}
RA = st.sidebar.button("Reglas de Asociacion")
MD = st.sidebar.button("Métricas de Distancia")
Cl = st.sidebar.button("Clustering")

#if True:
#    Asociacion.programa()

# from PIL.Image import init      ##No deben estar?
# import streamlit as st
# import MetricasDistancia            ##Bibliotecas (Otros archivos de Python, otras pestanas dentro del proyecto)
# import ReglasAsociacion
# import Bienvenida
# import ClusteringJerarquico
# import re

#Diccionario, como clave el nombre de lo que se muestra; como valor, la biblioteca
TABS = {"Métricas Distancia": Distancia, "Reglas de Asociación": Asociacion, "Clústering": Clustering}


# #Icono a la pestana y efectos de pestana
# st.set_page_config(page_icon=":brain:", page_title="IntelliShow", initial_sidebar_state="collapsed",)

# #Titulo de la barra izquierda de los algoritmos
st.sidebar.title('Algoritmos')
# #Anade botones a cada uno de los tabs, que son los algoritmos u otras pestanas
selection = st.sidebar.radio("Ir a", list(TABS.keys()))
page = TABS[selection]
page.programa() #Funcion definida en todas las bibliotecas 
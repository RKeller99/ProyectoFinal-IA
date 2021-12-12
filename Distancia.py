import streamlit as st


#1. Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, minkowski
from scipy.spatial import distance

def programa():
       
    ##colorFondo = '#FEFBF3'
    ##colorSecundarioFondo = "#F8F0DF"
    ##colorPrimario = '#79B4B7'
    
    st.header("Métricas de Distancia")
    met_dist=st.sidebar.selectbox('Selecciona la métrica que deseas trabajar:',('Euclidiana','Chebyshev', 'Manhattan', 'Minkowski'))
    if met_dist == 'Euclidiana':
        metrica = 'euclidean'
    if met_dist == 'Chebyshev':
        metrica = 'chebyshev'
    if met_dist == 'Manhattan':
        metrica = 'cityblock'
    if met_dist == 'Minkowski':
        metrica = 'minkowski'
        p_usr = st.sidebar.text_input("P", 1.5)

#2. Importar los datos desde un archivo .xlsx o .csv
    archivo = st.file_uploader("Seleccione el archivo", type = ['xlsx','csv'])

    #Los datos cuentan o no con encabezado
    physics=st.checkbox('Seleccionar si el encabezado forma parte de los datos')
    if physics == True:
        header_val = None
    else:
        header_val = 0

    #Cuando se seleccionó un archivo
    if archivo != None:
        Hipoteca = pd.read_csv(archivo, error_bad_lines=False, header=header_val)
        #Muestra los datos
        st.subheader("Tabla de datos")
        with st.expander("Desplegar tabla de datos"):
            colum = st.slider("¿Cuántas columnas de datos deseas observar?", min_value=1, max_value=len(Hipoteca), value = 10)
            st.write(Hipoteca.head(colum))

#3. Matrices de distancias
        if met_dist == 'Minkowski':
            MDst = cdist(Hipoteca, Hipoteca, metric=metrica, p = float(p_usr))
        else:
            MDst = cdist(Hipoteca, Hipoteca, metric=metrica) #Recibe matriz donde se realiza la observación
        ME = pd.DataFrame(MDst)
        st.subheader("Matriz de distancia")
        with st.expander("Desplegar matriz de distancia"):
            column = st.slider("¿Cuántas columnas de la matriz deseas observar?", min_value=1, max_value=len(ME), value = 10)
            st.write(ME.head(column))
        ##Descargar de matriz de distancia
        MDI = ME.to_csv().encode('utf-8')
        st.download_button('Descargar matriz de distancia', file_name='matriz_distancia_' + met_dist + '.csv', data = MDI, mime = 'text/csv')
        #Distancia entre objetos
        st.subheader("Distancia entre objetos")
        with st.expander("Desplegar opción distancia entre objetos"):

            col1, col2 = st.columns(2)
            Objeto1 = col1.slider("Selecciona Objeto 1: ", min_value=1, max_value=len(ME))
            Objeto2 = col2.slider("Seleccione Objeto 2: ", min_value=1, max_value=len(ME))
            if met_dist == 'Minkowski':
                st.write(distance.minkowski(Hipoteca.iloc[Objeto1], Hipoteca.iloc[Objeto2],p = float(p_usr)))
            if met_dist == 'Euclidiana':
                st.write(distance.euclidean(Hipoteca.iloc[Objeto1], Hipoteca.iloc[Objeto2]))
            if met_dist == 'Chebyshev':
                st.write(distance.chebyshev(Hipoteca.iloc[Objeto1], Hipoteca.iloc[Objeto2]))
            if met_dist == 'Manhattan':
                st.write(distance.cityblock(Hipoteca.iloc[Objeto1], Hipoteca.iloc[Objeto2]))

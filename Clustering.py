import streamlit as st


#1. Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#Para clustering jerárquico
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

#Para clustering particional
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator   

def programa():
       
    ##colorFondo = '#FEFBF3'
    ##colorSecundarioFondo = "#F8F0DF"
    ##colorPrimario = '#79B4B7'
    
    st.header("Clustering")
    tipo_clus = st.sidebar.selectbox('Selecciona el algoritmo que deseas trabajar:',('Particional','Jerárquico'))

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

#3. Selección de características
        #Mapa de calor, para observar las correlaciones
        CorrHipoteca = Hipoteca.corr(method='pearson')    #Medición se hace a nivel vectorial. Coeficiente de correlación de Pearson.
        #st.write(CorrHipoteca)
        fig = plt.figure(figsize=(14,7))
        MatrizInf = np.triu(CorrHipoteca)
        sns.heatmap(CorrHipoteca, cmap='RdBu_r', annot=True, mask=MatrizInf)
        st.subheader("Mapa de Calor")
        st.pyplot(fig)

        #Selección de características
        Characteristics = Hipoteca.columns
        st.sidebar.subheader("Selección de características")
        selection = st.sidebar.multiselect("Seleccionar características a utilizar", list(Characteristics))


        st.subheader("Matriz con selección de características finales")
        with st.expander("Desplegar matriz con selección de características finales"):
            columnas = st.slider("¿Cuántas columnas de la matriz deseas observar?", min_value=1, max_value=len(Hipoteca), value = 10)
            MatrizHipoteca = np.array(Hipoteca[selection])
            MHDF = pd.DataFrame(MatrizHipoteca)
            st.write(MHDF.head(columnas))

#3. Aplicación del Algoritmo
        st.sidebar.subheader("Método de estandarización")
        opciones_Estandarizacion = ['StandardScaler','MinMaxScaler']
        estandarizar_var = st.sidebar.radio("Seleccionar estandarización a utilizar", list(opciones_Estandarizacion))
        if estandarizar_var == 'StandardScaler':
            estandarizar = StandardScaler()
        if estandarizar_var == 'MinMaxScaler':    
            estandarizar = MinMaxScaler()
        MEstandarizada = estandarizar.fit_transform(MatrizHipoteca) 
        st.subheader("Matriz Estandarizada")
        with st.expander("Desplegar matriz estandarizada"):
            columnas = st.slider("¿Cuántas columnas de la matriz estandarizada deseas observar?", min_value=1, max_value=len(MEstandarizada), value = 10)
            MHE = pd.DataFrame(MEstandarizada)
            st.write(MHE.head(columnas))

        if tipo_clus == 'Jerárquico':
            #Gráfica clusters jerárquicos
            fig = plt.figure(figsize=(10, 7))
            plt.title("Casos de hipoteca")
            plt.xlabel('Hipoteca')
            plt.ylabel('Distancia')
            Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))
            plt.axhline(y=5.4, color='orange', linestyle='--')
            st.subheader("Gráfica de Clustering Jerárquico")
            st.pyplot(fig)

            #Se crean las etiquetas de los elementos en los clústeres
            st.sidebar.subheader("Número de clusters para análisis")
            nclusters = st.sidebar.text_input("Indique el número de clusters", 7)
            MJerarquico = AgglomerativeClustering(n_clusters=int(nclusters), linkage='complete', affinity='euclidean')
            MJerarquico.fit_predict(MEstandarizada)
            Hipoteca = Hipoteca[selection]
            Hipoteca['clusterH'] = MJerarquico.labels_
            st.subheader("Matriz con clusters asignados")
            with st.expander("Desplegar matriz con clusters asignados"):
                columnascca = st.slider("¿Cuántas columnas de la matriz con clusters deseas observar?", min_value=1, max_value=len(MEstandarizada), value = 10)
                MHCCA = pd.DataFrame(Hipoteca)
                st.write(MHCCA.head(columnascca))
            st.subheader("Número de elementos por cada cluster")
            with st.expander("Número de elementos por cada cluster"):
                st.write(Hipoteca.groupby(['clusterH'])['clusterH'].count())
            st.subheader("Análisis por cluster seleccionado")
            with st.expander("Análisis por cluster"):
                st.sidebar.subheader("Análisis por cluster seleccionado")
                nclusteranalysis = st.sidebar.slider("Indique el cluster que desea analizar", min_value=0, max_value=int(nclusters))
                st.write(Hipoteca[Hipoteca.clusterH == int(nclusteranalysis)])
            
            
            st.subheader("Análisis por centroides de cada cluster")
            with st.expander("Tabla de centroides"):
                CentroidesH = Hipoteca.groupby('clusterH').mean()
                st.write(CentroidesH)

            with st.expander("Análisis final de todos los clusters"):
                st.subheader("Número de clusters: " + nclusters)
                AnalisisClusters = CentroidesH.values.tolist()
                for contador in range (0, int(nclusters)):
                    st.markdown("__Cluster__ " + "__"+ str(contador) +"__" + ":")
                    contador2 = 0
                    for item in CentroidesH:
                        st.write(item + ' :', AnalisisClusters[contador][contador2])
                        contador2+=1






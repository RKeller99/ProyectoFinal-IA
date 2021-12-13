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
        if MatrizHipoteca.size != 0:
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
                MClustering = AgglomerativeClustering(n_clusters=int(nclusters), linkage='complete', affinity='euclidean')
                MClustering.fit_predict(MEstandarizada)
                Hipoteca = Hipoteca[selection]            
            
            if tipo_clus == 'Particional':
                #Gráfica clusters particional
                SSE = []
                for i in range(2, 12):
                    km = KMeans(n_clusters=i, random_state=0) #Valor aleatorio a partir de hora del sistema.
                    km.fit(MEstandarizada)                    #Estimación con respecto a matriz de datos estandarizados
                    SSE.append(km.inertia_)                   #Que nos devuelva la inercia para poder graficar

                #Se grafica SSE en función de k
                fig = plt.figure(figsize=(10, 7))
                plt.plot(range(2, 12), SSE, marker='o')     #Se grafican valores finales de K, mediante puntos.
                plt.xlabel('Cantidad de clusters *k*')
                plt.ylabel('SSE')
                plt.title('Elbow Method')
                st.subheader("Gráfica de Método del Codo")
                st.pyplot(fig)

                # kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
                # kl.elbow
                # plt.style.use('ggplot')
                # fig = kl.plot_knee()
                # st.subheader("Gráfica de Método del Codo")
                # st.pyplot(fig)      

                #Se crean las etiquetas de los elementos en los clústeres
                st.sidebar.subheader("Número de clusters para análisis")
                nclusters = st.sidebar.text_input("Indique el número de clusters", 4)
                Hipoteca = Hipoteca[selection]
                MClustering = KMeans(n_clusters=int(nclusters), random_state=0).fit(MEstandarizada)  #Se define el número de clusters(4)
                MClustering.predict(MEstandarizada)
                Hipoteca = Hipoteca[selection]           


            Hipoteca['cluster'] = MClustering.labels_
            st.subheader("Matriz con clusters asignados")
            with st.expander("Desplegar matriz con clusters asignados"):
                columnascca = st.slider("¿Cuántas columnas de la matriz con clusters deseas observar?", min_value=1, max_value=len(MEstandarizada), value = 10)
                MHCCA = pd.DataFrame(Hipoteca)
                st.write(MHCCA.head(columnascca))
            st.subheader("Número de elementos por cada cluster")
            with st.expander("Número de elementos por cada cluster"):
                elementos_por_cluster = Hipoteca.groupby(['cluster'])['cluster'].count()
                st.write(elementos_por_cluster)
            st.subheader("Análisis por cluster seleccionado")
            with st.expander("Análisis por cluster"):
                st.sidebar.subheader("Análisis por cluster seleccionado")
                nclusteranalysis = st.sidebar.slider("Indique el cluster que desea analizar", min_value=0, max_value=int(nclusters)-1)
                st.write(Hipoteca[Hipoteca.cluster == int(nclusteranalysis)])
            
            
            st.subheader("Análisis por centroides de cada cluster")
            with st.expander("Tabla de centroides"):
                Centroides = Hipoteca.groupby('cluster').mean()
                st.write(Centroides)

            st.subheader("Análisis final de todos los clusters")
            st.markdown("Número de clusters: " + nclusters)
            AnalisisClusters = Centroides.values.tolist()
            Comentarios = []
            Descarga = "########################################################\n\n"
            Descarga += "Análisis clusters en archivo: " + archivo.name + "\n\n"
            Descarga += "Algoritmo: " + "Clustering" + tipo_clus + "\n"
            Descarga += "Método de estandarización: " + estandarizar_var + "\n\n"
            Descarga += "########################################################\n\n"
            for contador in range (0, int(nclusters)):
                st.markdown("__Cluster__ " + "__"+ str(contador) +"__" + ":")
                Descarga += "Cluster "+ str(contador) + ":\n"
                contador2 = 0
                st.write("Número de elementos: " + str(elementos_por_cluster[contador]))
                Descarga += "Número de elementos: " + str(elementos_por_cluster[contador]) + "\n"
                for item in Centroides:
                    st.write(item + ' :', AnalisisClusters[contador][contador2])
                    Descarga += item + ' :' + str(AnalisisClusters[contador][contador2]) + "\n"
                    contador2+=1
                Comentarios.append(st.text_area('Comentarios cluster ' + str(contador) + ' :', ""))
                Descarga += "Comentarios:\n"
                Descarga += Comentarios[contador]
                Descarga += "\n\n\n"
            st.download_button('Descargar análisis de clusters', file_name='cluster_' + tipo_clus +'.txt', data = Descarga)      
import threading
import streamlit as st


#1. Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Para árbol de decisión - Pronóstico
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection

#Para árbol de decisión - Clasificación
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection

#Gráficas de los árboles
import graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from sklearn.tree import export_text

def programa():
       
    ##colorFondo = '#FEFBF3'
    ##colorSecundarioFondo = "#F8F0DF"
    ##colorPrimario = '#79B4B7'
    
    st.header("Árboles de Decisión")
    tipo_arbol = st.sidebar.selectbox('Selecciona el tipo de árbol con el que deseas trabajar:',('Pronóstico','Clasificación'))

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
        Archivo = pd.read_csv(archivo, error_bad_lines=False, header=header_val)
        #Muestra los datos
        st.subheader("Tabla de datos")
        with st.expander("Desplegar tabla de datos"):
            colum = st.slider("¿Cuántas columnas de datos deseas observar?", min_value=1, max_value=len(Archivo), value = 10)
            st.write(Archivo.head(colum))

#2. Gráfica de características
        # if tipo_arbol == 'Pronóstico':
        #     st.subheader("Gráfica de características")
        #     with st.expander("Desplegar gráfica de datos"):
        #         st.sidebar.subheader("Primera Gráfica de Análisis")
        #         xVar = st.sidebar.radio("Seleccione la variable X para su primera gráfica de análisis de datos: ", Archivo.columns)
        #         yVar = st.sidebar.radio("Seleccione la variable Y para su primera gráfica de análisis de datos: ", Archivo.columns)
        #         fig = plt.figure(figsize=(20, 5))
        #         plt.plot(Archivo[xVar], Archivo[yVar], color='green', marker='o', label='Area')
        #         plt.xlabel(xVar)
        #         plt.ylabel(yVar)
        #         plt.title('Primera Gráfica de Análisis')
        #         plt.grid(True)
        #         plt.legend()
        #         st.pyplot(fig)

#3. Selección de características
        #Mapa de calor, para observar las correlaciones
        ArchivoCorrelacion = Archivo.corr(method='pearson')    #Medición se hace a nivel vectorial. Coeficiente de correlación de Pearson.
        #st.write(CorrHipoteca)
        fig = plt.figure(figsize=(14,7))
        MatrizInf = np.triu(ArchivoCorrelacion)
        sns.heatmap(ArchivoCorrelacion, cmap='RdBu_r', annot=True, mask=MatrizInf)
        st.subheader("Mapa de Calor")
        st.pyplot(fig)

        #Selección de características
        Characteristics = Archivo.columns
        st.sidebar.subheader("Selección de características")
        selection = st.sidebar.multiselect("Seleccionar características a utilizar", list(Characteristics))

        st.subheader("Matriz con selección de características finales")
        with st.expander("Desplegar matriz con selección de características finales"):
            columnas = st.slider("¿Cuántas columnas de la matriz deseas observar?", min_value=1, max_value=len(Archivo), value = 10)
            MatrizArchivo = np.array(Archivo[selection])
            MADF = pd.DataFrame(MatrizArchivo)
            MADF.columns = selection
            st.write(MADF.head(columnas))

#3. Definición de variables predictoras y de clase
        st.subheader("Definición de variables predictoras y de clase")        

        #Variables predictoras (6 variables, podemos usar nombre del campo o posición de las variables)
        CharacteristicsMADF = MADF.columns
        st.sidebar.subheader("Selección variables predictoras")
        selectionVP = st.sidebar.multiselect("Seleccionar variables predictoras a utilizar", list(CharacteristicsMADF))

        st.markdown("__Matriz con variables predictoras__")
        with st.expander("Desplegar matriz con variables predictoras"):
            columnasVP = st.slider("¿Cuántas columnas de la matriz de variables predictoras deseas observar?", min_value=1, max_value=len(Archivo), value = 10)
            X = np.array(MADF[selectionVP])
            xMuestra = pd.DataFrame(X)
            xMuestra.columns = selectionVP
            st.write(xMuestra.head(columnasVP))


        #Variable clase
        CharacteristicsMADFVC = MADF.columns
        st.sidebar.subheader("Selección variable de clase")
        selectionVC = st.sidebar.multiselect("Seleccionar variable de clase a utilizar", list(CharacteristicsMADF))

        st.markdown("__Matriz con variables de clase__")
        with st.expander("Desplegar matriz con variables de clase"):
            columnasVC = st.slider("¿Cuántas columnas de la matriz de variables de clase deseas observar?", min_value=1, max_value=len(Archivo), value = 10)        
            Y = np.array(MADF[selectionVC])
            yMuestra = pd.DataFrame(Y)
            yMuestra.columns = selectionVC
            st.write(yMuestra.head(columnasVC))

        st.sidebar.subheader("Parámetros para división de datos")
        test_size_usr = st.sidebar.text_input("Test Size", 0.2)
        random_state_usr = st.sidebar.text_input("Random State", 1234)
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = float(test_size_usr), random_state = int(random_state_usr), shuffle = True)  
        
        st.sidebar.subheader("Selección de Parámetros - Aplicación del Algoritmo")
        profundidad_max_usr = st.sidebar.text_input("Profundidad Máxima", 8)
        min_sample_split_usr = st.sidebar.text_input("Mínimo muestras separación", 4)
        min_sample_leaf_usr = st.sidebar.text_input("Mínimo muestras hoja", 2)
        if tipo_arbol == 'Pronóstico':
            Arbol = DecisionTreeRegressor(max_depth=int(profundidad_max_usr), min_samples_split=int(min_sample_split_usr), min_samples_leaf=int(min_sample_leaf_usr))
        if tipo_arbol == 'Clasificación':
            Arbol = DecisionTreeClassifier(max_depth=int(profundidad_max_usr), min_samples_split=int(min_sample_split_usr), min_samples_leaf=int(min_sample_leaf_usr))
        Arbol.fit(X_train, Y_train)
        Y_Pronostico = Arbol.predict(X_train)

#5.Validación del Algoritmo
        st.subheader("Validación del Algoritmo")
        if tipo_arbol == 'Pronóstico':
            st.write('Criterio: \n', Arbol.criterion)
            st.write("MAE - Error Absoluto Medio: %.4f" % mean_absolute_error(Y_train, Y_Pronostico))
            st.write("MSE - Error Cuadrático Medio: %.4f" % mean_squared_error(Y_train, Y_Pronostico))
            st.write("RMSE - Raíz Cuadrada del Error Cuadrático Medio: %.4f" % mean_squared_error(Y_train, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
            st.write('Score: %.4f' % r2_score(Y_train, Y_Pronostico))
        if tipo_arbol == 'Clasificación':
            Y_Clasificacion = Arbol.predict(X_validation)
            Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), Y_Clasificacion, rownames=['Real'], colnames=['Clasificación']) 
            Matriz_Clasificacion
            st.write('Criterio: \n', Arbol.criterion)
            #st.write(classification_report(Y_validation, Y_Clasificacion))
        st.write("Exactitud", Arbol.score(X_validation, Y_validation)*100, " % ")
        st.markdown("__Importancia de las variables__")
        Importancia = pd.DataFrame({'Variable': list(Archivo[selectionVP]), 'Importancia': + Arbol.feature_importances_}).sort_values('Importancia', ascending=False)         
        st.write(Importancia)

#6. Conformación del modelo de Pronóstico
        st.subheader("Conformación del modelo de pronóstico")
        #Elementos = export_graphviz(Arbol, feature_names = selectionVP)  
        #Arbol1 = graphviz.Source(Elementos)
        st.markdown("__Árbol de Decisión__")
        Arbol2 = plt.figure(figsize=(16,16))  
        plot_tree(Arbol, feature_names = selectionVP)
        st.pyplot(Arbol2)
        plt.savefig('arbol_de_decision_' + tipo_arbol +'.png')
        with open('arbol_de_decision_' + tipo_arbol +'.png',"rb") as file:
            button = st.download_button("Descarga del árbol de decisión", data = file, file_name = 'arbol_de_decision_' + tipo_arbol +'.png', mime ="image/png")
        #st.download_button("Descarga árbol de decisión", data = Arbol1.pipe(format='svg'), file_name = "arbol_de_decision.svg", mime ="image/svg")  
        st.markdown("__Esquema del Árbol de Decisión__")
        with st.expander("Desplegar esquema del árbol de decisión"):
            Reporte = export_text(Arbol, feature_names = selectionVP)
            Reporte = Reporte.split("\n")
            Esquema = "########################################################\n\n"
            Esquema += "Esquema árbol de decisión en archivo: " + archivo.name + "\n\n"
            Esquema += "Tipo de árbol: " +  tipo_arbol + "\n\n"
            Esquema += "Variables Predictoras:\n"
            for elemento in selectionVP:
                Esquema += elemento + ", "
            Esquema += "\n\nVariable Clase:\n"
            for elemento in selectionVC:
                Esquema += elemento + "\n\n"
            Esquema += "########################################################\n\n"
            for elemento in Reporte:
                st.text(elemento)
                Esquema += elemento + "\n"
        st.download_button('Descargar reporte escrito del árbol de decisión', file_name='esquema_arbol_' + tipo_arbol+ '.txt', data = Esquema)    

#7. Nuevas Predicciones
        st.subheader("Nuevas Predicciones") 
        l = []
        for k in range (0, len(selectionVP)) :
            l.append(st.text_input(selectionVP[k], 0))
        st.markdown("__Valor de la nueva predicción:__")
        TextoPrediccion = ""
        for k in range (0, len(selectionVP)) :
            if k == len(selectionVP)-1: 
                TextoPrediccion += "" + str(l[k]) +""
            else:
                TextoPrediccion += "" + str(l[k]) +", "
        NuevaPrediccion = pd.DataFrame(x.split(',') for x in TextoPrediccion.split('\n'))
        #st.write(NuevaPrediccion)
        arr = Arbol.predict(NuevaPrediccion)
        st.write(arr[0])
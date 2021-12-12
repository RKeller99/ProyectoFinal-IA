from os import sep
import streamlit as st


#1. Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler


#Para regresión Logística
from sklearn import linear_model                    
from sklearn import model_selection                 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix        
from sklearn.metrics import accuracy_score          

def programa():
       
    ##colorFondo = '#FEFBF3'
    ##colorSecundarioFondo = "#F8F0DF"
    ##colorPrimario = '#79B4B7'
    st.header("Regresión Logística")

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

#3. Selección de características
        st.subheader("Selección de características") 
        #Mapa de calor, para observar las correlaciones
        CorrelacionArchivo = Archivo.corr(method='pearson')    #Medición se hace a nivel vectorial. Coeficiente de correlación de Pearson.
        #st.write(CorrHipoteca)
        fig = plt.figure(figsize=(14,7))
        MatrizInf = np.triu(CorrelacionArchivo)
        sns.heatmap(CorrelacionArchivo, cmap='RdBu_r', annot=True, mask=MatrizInf)
        st.markdown("__Mapa de Calor__")
        st.pyplot(fig)

        #Selección de características
        Characteristics = Archivo.columns
        st.sidebar.subheader("Selección de características")
        selection = st.sidebar.multiselect("Seleccionar características a utilizar", list(Characteristics))


        st.markdown("__Matriz con selección de características finales__")
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

#4.Aplicación del Algoritmo
        st.subheader("Aplicación del Algoritmo")
        test_size_usr = st.sidebar.text_input("Test Size", 0.2)
        random_state_usr = st.sidebar.text_input("Random State", 1234)
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = float(test_size_usr), random_state = int(random_state_usr), shuffle = True)  
        Clasificacion = linear_model.LogisticRegression()
        Clasificacion.fit(X_train, Y_train)

        # #Predicciones para tomar un valor u otro
        # Probabilidad = Clasificacion.predict_proba(X_validation)
        # st.markdown("?__Predicciones probabilísticas para tomar cualquiera de dos valores__")
        # with st.expander("Desplegar matriz con predicciones probabilísticas para tomar cualquiera de dos valores"):
        #     columnasPDV = st.slider("¿Cuántas columnas de la matriz de probabilidades de dos valores deseas observar?", min_value=1, max_value=len(Archivo), value = 10)        
        #     DFProbabilidad = pd.DataFrame(Probabilidad)
        #     st.write(DFProbabilidad.head(columnasPDV))   

        #Predicciones finales
        Predicciones = Clasificacion.predict(X_validation)
        st.markdown("__Predicciones con clasificación final__")
        with st.expander("Desplegar matriz con predicciones finales"):
            columnasMPF = st.slider("¿Cuántas columnas de la matriz de predicciones finales deseas observar?", min_value=1, max_value=len(Archivo), value = 10)        
            DFPredicciones = pd.DataFrame(Predicciones)
            DFPredicciones.columns = ['Predicción Final']
            st.write(DFPredicciones.head(columnasMPF))   

#5.Validación del Modelo
        st.subheader("Validación del modelo")  
        Y_Clasificacion = Clasificacion.predict(X_validation)
        Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), Y_Clasificacion, rownames=['Real'], colnames=['Clasificación'])
        st.markdown("__Matriz de clasificación__") 
        st.write(Matriz_Clasificacion)
        st.write("Exactitud", Clasificacion.score(X_validation, Y_validation)*100, " % ")
        #st.markdown("__Reporte de Clasificación__") 
        #st.write(classification_report(Y_validation, Y_Clasificacion))        

#6.Ecuación del modelo de clasificación
        st.subheader("Ecuación del modelo de clasificación") 
        top = len(selectionVP)
        CoeficientesMod = Clasificacion.coef_.tolist()
        Intercept = Clasificacion.intercept_.tolist()
        Ecuacion = "Prob = 1/1+𝑒^−(𝑎+𝑏𝑋))\n"
        st.write("Prob = 1/1+𝑒^−(𝑎+𝑏𝑋))")
        st.write("a+bX = " + str(Intercept[0]))
        Ecuacion += "a+bX = " + str(Intercept[0]) + "\n"
        for i in range (0, top-1) :
            if float(CoeficientesMod[0][i]) < 0:
                st.write( str(CoeficientesMod[0][i]) + "[" + str(selectionVP[i]) + "]")
                Ecuacion += str(CoeficientesMod[0][i]) + "[" + str(selectionVP[i]) + "]\n"
            else:
                st.write( "+" + str(CoeficientesMod[0][i]) + "[" + str(selectionVP[i]) + "]")
                Ecuacion +=  "+" + str(CoeficientesMod[0][i]) + "[" + str(selectionVP[i]) + "]\n"
        
        Descarga = "########################################################\n\n"
        Descarga += "Regresión Logística en archivo: " + archivo.name + "\n\n"
        Descarga += "Variables Predictoras:\n"
        for elemento in selectionVP:
            Descarga += elemento + ", "
        Descarga += "\n\nVariable Clase:\n"
        for elemento in selectionVC:
            Descarga += elemento + "\n\n"
        Descarga += "Test Size: " + test_size_usr + "\n"
        Descarga += "Random State: " + random_state_usr + "\n\n"
        Descarga += "########################################################\n\n"
        st.download_button('Descargar modelo regresión logística', file_name='modelo_regresion_logistica.txt', data = Descarga + Ecuacion)


            
#7. Nuevas Predicciones
        st.subheader("Nuevas Predicciones") 
        NuevaPrediccionTx = "\n########################################################\n\n"
        NuevaPrediccionTx += "Nueva Predicción: \n\n"
        l = []
        for k in range (0, len(selectionVP)) :
            l.append(st.text_input(selectionVP[k], 0))
            NuevaPrediccionTx += str(selectionVP[k]) + ": " + str(l[k]) + "\n"
        st.markdown("__Valor de la nueva predicción:__")
        TextoPrediccion = ""
        for k in range (0, len(selectionVP)) :
            if k == len(selectionVP)-1: 
                TextoPrediccion += "" + str(l[k]) +""
            else:
                TextoPrediccion += "" + str(l[k]) +", "
        NuevaPrediccion = pd.DataFrame(x.split(',') for x in TextoPrediccion.split('\n'))
        #st.write(NuevaPrediccion)
        arr = Clasificacion.predict(NuevaPrediccion)
        st.write(arr[0])
        NuevaPrediccionTx += "\n7Valor de la nueva predicción: " + str(arr[0]) + "\n\n"
        NuevaPrediccionTx += "########################################################\n\n"
        st.download_button('Descargar nueva predicción modelo regresión logística', file_name='nueva_prediccion_regresion_logistica.txt', data = Descarga + Ecuacion + NuevaPrediccionTx)
        
import streamlit as st


#1. Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori



def programa():
       
    ##colorFondo = '#FEFBF3'
    ##colorSecundarioFondo = "#F8F0DF"
    ##colorPrimario = '#79B4B7'
    
    st.header("Reglas de Asociación")


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
        DataFrameArchivo = pd.read_csv(archivo, error_bad_lines=False, header=header_val)
        #Muestra los datos
        st.subheader("Tabla de datos")
        with st.expander("Desplegar tabla de datos"):
            colum = st.slider("¿Cuántas columnas de datos deseas observar?", min_value=1, max_value=len(DataFrameArchivo), value = 10)
            st.write(DataFrameArchivo.head(colum))

#3. Procesamiento de Datos/Exploración de los Items
        Transacciones = DataFrameArchivo.values.reshape(-1).tolist() #Pasamos todos los datos a una lista
        Lista = pd.DataFrame(Transacciones)
        Lista['Frecuencia'] = 1  # Valor que después se reemplazará, es nada más para agregar la columna. 
        Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=False)
        Lista['Porcentaje [%]'] = (Lista['Frecuencia']/Lista['Frecuencia'].sum()*100)
        Lista = Lista.rename(columns={0:'Item'})

        st.subheader("Tabla de frecuencia de datos")
        st.write(Lista)

        #Gráfica de frecuencia de datos
        fig = plt.figure(figsize=(16,16), dpi = 300)
        plt.ylabel('Item')
        plt.xlabel('Frecuencia')
        #fig.patch.set_facecolor(colorFondo)
        ax = plt.axes()
        #ax.set(facecolor = colorSecundarioFondo)
        plt.barh(Lista['Item'], width=Lista['Frecuencia']) #, color=colorPrimario

        st.subheader("Generación de gráfica de frecuencia de datos")
        st.pyplot(fig)
        plt.savefig('grafica_frecuencia_datos.png')
        with open("grafica_frecuencia_datos.png","rb") as file:
            button = st.download_button("Descarga de la gráfica", data = file, file_name = "grafica_frecuencia_datos.png", mime ="image/png")


#4. Aplicación del algoritmo
        #Selección de parámetros
        st.sidebar.subheader("Selección de Parámetros - Aplicación del Algoritmo")
        col1, col2, col3 = st.sidebar.columns(3)
        soporte = col1.text_input("Soporte", 0.01)
        confianza = col2.text_input("Confianza", 0.3)
        elevacion = col3.text_input("Elevación", 2.0)


        #Aplicación del algoritmo
        Lista = DataFrameArchivo.stack().groupby(level=0).apply(list).tolist()
        ReglasC1 = apriori(Lista, min_support=float(soporte), min_confidence=float(confianza), min_lift=float(elevacion))
        ResultadoC1 = list(ReglasC1)
        st.subheader("Número de reglas de asociación: " + str(len(ResultadoC1)))

        j = 1
        Comentarios = []
        Descarga = "########################################################\n\n"
        Descarga += "Reglas de Asociación en archivo: " + archivo.name + "\n\n"
        Descarga += "Parámetros en Aplicación del algoritmo:\n"
        Descarga += "Soporte: " + soporte + "\nConfianza: " + confianza + "\nElevación: " + elevacion + "\n\n"
        Descarga += "########################################################\n\n"
        for item in ResultadoC1:
            Lista = list(item[0])
            Objetos = "Objetos en la regla:"
            st.markdown("__Regla__ " + "__"+str(j)+"__" + ":")
            Descarga += "Regla " + str(j) + " :"
            for i in range (0, len(Lista)):
                if i == len(Lista)-1:
                    Objetos += " " + str(Lista[i])
                    Descarga += " " + str(Lista[i])
                else:
                    Objetos += " " + str(Lista[i] + ",")
                    Descarga += " " + str(Lista[i] + ",")
            st.write(Objetos)
            st.write("Soporte: ", round(item[1],5), ", Confianza: ", round(list(item[2][0])[2],5), ", Elevación: ", round(list(item[2][0])[3],5))
            Descarga += "\n" + "Soporte: " + str(round(item[1],5)) + ", Confianza: " + str(round(list(item[2][0])[2],5)) + ", Elevación: " + str(round(list(item[2][0])[3],5)) + "\n"
            Comentarios.append(st.text_area('Comentarios regla ' + str(j) + ' :', ""))
            Descarga += "Comentarios:\n"
            Descarga += Comentarios[j-1]
            Descarga += "\n\n\n"
            j+=1
        st.download_button('Descargar reglas de asociación', file_name='reglas_asociacion.txt', data = Descarga)
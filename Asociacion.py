from numpy.lib.shape_base import column_stack
import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori



def programa():
       
    ##colorFondo = '#FEFBF3'
    ##colorSecundarioFondo = "#F8F0DF"
    ##colorPrimario = '#79B4B7'
    
    st.title("Reglas de Asociación")

    archivo = st.file_uploader("Seleccione el archivo", type = ['xlsx','csv'])
    if archivo != None:
        st.write("Archivo: " + "*"+archivo.name+archivo.type+"*")

        DataFrameArchivo = pd.read_csv(archivo, error_bad_lines=False)

        st.write(DataFrameArchivo)



        Transacciones = DataFrameArchivo.values.reshape(-1).tolist()
        

        Lista = pd.DataFrame(Transacciones)
        Lista['Frecuencia'] = 1  # valor que después se reemplazará, es nada más para agregar la columna. 
        Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True)
        Lista['Porcentaje'] = (Lista['Frecuencia']/Lista['Frecuencia'].sum())
        Lista = Lista.rename(columns={0:'Item'})
        st.write(Lista)


        









        fig = plt.figure(figsize=(16,16), dpi = 300)
        plt.ylabel('Item')
        plt.xlabel('Frecuencia')
        #fig.patch.set_facecolor(colorFondo)
        ax = plt.axes()
        #ax.set(facecolor = colorSecundarioFondo)
        plt.barh(Lista['Item'], width=Lista['Frecuencia']) #, color=colorPrimario
        st.pyplot(fig)


        col1, col2, col3 = st.columns(3)
        soporte = col1.text_input("Soporte", 0.02)
        confianza = col2.text_input("Confianza", 0.02)
        elevacion = col3.text_input("Elevación", 1.2)



        Lista = DataFrameArchivo.stack().groupby(level=0).apply(list).tolist()
        ReglasC1 = apriori(Lista, 
                   min_support=float(soporte), 
                   min_confidence=float(confianza), 
                   min_lift=float(elevacion))
        ResultadoC1 = list(ReglasC1)
        st.subheader("Reglas de asociación encontradas: " + str(len(ResultadoC1))  )

        j = 0
        for i in ResultadoC1:
            j+=1
            st.markdown("__Regla__ " + "__"+str(j)+"__" + ":")
            st.write(list(i[0]), "Soporte: ", round(i[1],5), ", Confianza: ", round(list(i[2][0])[2],5), ", Elevación: ", round(list(i[2][0])[3],5))
            st.text_area("Observaciones", key=j)
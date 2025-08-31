# Proceso del proyecto (26/08 - 31/08)

## 26/08

Tras un primer vistazo rápido, se nota la falta de datos en varias, columnas, así como patrones de repetición en algunas columnas. Por ello, comienzo con el primer notebook del proyecto, "01_fuentes.ipynb"  
 
Tras ejecutar, la carga de datos, exploración de estructura y el análisis de la calidad de datos, se confirman las primeras sospechas. Mediante las visualizaciones iniciales puedo observar con más detalle y de forma más gráfica, la realidad de los datos que contiene nuestro archivo original.  

Adapto el archivo para obtener una comprensión más dinámica de los resultados, enfocándome en la adquisición segura de los datos y la exploración inicial sistemática, estableciendo las bases para el análisis posterior. 

## 27/08

Tras un periodo de observación y meditación inicial, procedo con la limpieza y optimización de nuestros datos, "02_limpieza_eda.ipynb". Mediante el código de este archivo, pretendo implementar una limpieza robusta de datos y un análisis exploratorio avanzado mediante clases especializadas.  

La primera de ellas, "DataCleaner" (en sección de Limpieza de datos), encapsula todo el proceso de limpieza de datos de manera sistemática. Dentro de ella existen diversos métodos con diferentes utilidades, debido a la complejidad (o suciedad) inicial de los datos del archivo .csv original. 

## 28/08

Dedico el día a mejorar la visibilidad y comprensión de todo el código desarrollado hasta el momento, así como los outputs de cada celda, tratando de encontrar también posibles errores. 

## 29/08

Tras terminar con la limpieza de datos, comienzo con el entrenamiento y la evaluación de modelos en el archivo "03_entrenamiento.ipynb". Aquí se implementa un sistema comppleto de entrenamiento y evaluación de modelos de Machine Learning para clasificar respuestas de empresas a quejas financieras.  

Al igual que en anteriores notebooks, decido implementar múltiples clases y funciones o métodos para automatizar totalmente cualquier toma de decisión posible en base a los resultados oobtenidos. 

Durante las primeras pruebas del código me encuentro con el problema que el método de entrenamiento SVC, tarda más de 6 horas en ejecutarse. Debido a ello, y viendo que no afecta en absoluto a la conclusión de que Random Forest es el mejor método, decido eliminar la sección dedicada al SVC dentro de la clase "ModelTrainer", más concretamente del método "define_models". De haberlo mantenido, hubiese ralentizado en exceso las futuras pruebas afectando así a posteriores fases de limpieza, correcta documentación y efectividad del proyecto.  

## 30/08

Sigo adelante con el archivo "03_entrenamiento.ipynb", buscando total transparencia, reproducibilidad y escalabilidad en el código y los resultados. En todo momento, busco una estructura bien organizada y documentada, con un manejo conmprensivo de diferentes escenarios, y una evaluación rigurosa y multifacética.  

Mediante este código, se permite la automatización de clasificación de quejas, así como la mejora en eficiencia de respuesta, reflejando así una commprensión sólida tanto de principios técnicos de Machine Learning, como de consideraciones prácticas en entornos empresariales. Durante la ejecución de "Optimización de los Hiperparámetros", el equipo se me ha bloqueado y hasta reiniciado en varias ocasiones, por lo que decido alterar algunos parámetros (cambios comentados en el código, de cara a posible futura implementación) para asi poder garantizar la ejecución del archivo.  

Tras conseguir el archivo "prediction.py", decido ponerlo a prueba en el notebook "04_prueba_prediction.ipynb". El output resulta plano o simple, por lo que decido dejar para el día siguiente la posibilidad de mejorar la visibilidad de los resultados. Para no perder el hilo del momento, decido revisar todo el código del día para asegurar la precisión de la correspondiente documentación.

## 31/08

Último día, decido dedicarlo a la ya mencionada implementación visual del output de las predicciones, más concretamente en el notebook "05_prueba_mejorada". Además adapto el código para posibles escenarios no contemplados anteriormente. Antes de subir todo el proyecto, realizaré una copia y volveré a ejecutar todo el proceso para así garantizar la ausencia de errores que haya podido pasar por alto.

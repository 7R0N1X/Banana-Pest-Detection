##  Detección de ceramidia viridis en hojas de plátano
Este proyecto busca detectar la plaga ceramidia viridis en hojas de plátano mediante el uso de aprendizaje automático. 

### Requisitos
- opencv-python
- numpy
- sklearn
- matplotlib

### Uso
- Asegurarse de tener las imágenes de las plantas infectadas y sanas en una carpeta llamada 'dataset/pests/ceramidia-viridis/' y 'dataset/pests/healthy-leaves/' respectivamente.
- Ejecutar el script con python.
- El script cargará las imágenes, las redimensionará a 720x720 píxeles y extraerá características de las imágenes para entrenar el modelo.
- El script también mostrará el número de imágenes cargadas y la matriz de confusión del modelo.

### Personalización
- Para detectar plagas diferentes, es necesario proporcionar imágenes de plagas y hojas sanas correspondientes y ajustar el código para extraer características adecuadas.
- El tamaño de las imágenes puede ser ajustado según las necesidades.
- El algoritmo de clasificación puede ser reemplazado por otro algoritmo de aprendizaje automático.
- Puede ser necesario aumentar el tamaño del conjunto de datos de entrenamiento para mejorar el rendimiento del modelo.

### Limitaciones
- El rendimiento del modelo puede ser limitado si el conjunto de datos de entrenamiento no es suficientemente grande o variado.
- El rendimiento del modelo también puede ser afectado por la calidad de las imágenes y las características extraídas.

Este código es solo un ejemplo básico de cómo se puede utilizar el aprendizaje automático para detectar plagas en plantas. Se recomienda investigar más sobre el tema y experimentar con diferentes conjuntos de datos y algoritmos para obtener mejores resultados.
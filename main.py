import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Carga dos listas de imágenes, una con imágenes infectadas y otra con imágenes sanas.
    creamidia_viridis = [f'dataset/ceramidia-viridis/{img}' for img in os.listdir('dataset/ceramidia-viridis')]
    healthy_leaves = [f'dataset/healthy-leaves/{img}' for img in os.listdir('dataset/healthy-leaves')]

    # Inicializar listas vacías para almacenar imágenes infectadas y sanas.
    infected_images = []
    healthy_images = []
    # Recorre cada archivo de imagen en la lista de creamidia_viridis.
    for image_file in creamidia_viridis:
        # Lee la imagen del archivo.
        image = cv2.imread(image_file)
        # Agrega la imagen a la lista de imágenes infectadas.
        infected_images.append(image)
    # Recorre cada archivo de imagen en la lista de healthy_leaves.
    for image_file in healthy_leaves:
        # Lee la imagen del archivo.
        image = cv2.imread(image_file)
        # Agrega la imagen a la lista de imágenes sanas.
        healthy_images.append(image)
    print('************************* {} *************************'.format('Total de imágenes cargadas'))
    print(f'Imágenes infectadas: {len(infected_images)}')
    print(f'Imágenes sanas: {len(healthy_images)}')

    # Redimensiona las imágenes de plantas infectadas y sanas a un tamaño de 720x720 píxeles.
    for i in range(len(infected_images)):
        infected_images[i] = cv2.resize(infected_images[i], (720, 720))
    for i in range(len(healthy_images)):
        healthy_images[i] = cv2.resize(healthy_images[i], (720, 720))

    # Extrae características de las imágenes de plantas infectadas y sanas.
    def extract_features(img):
        # Convertir a escala de grises.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.dilate(gray, None, iterations=1)
        # Aplicar umbral para destacar huecos.
        _, threshold = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        # Encontrar contornos de los huecos.
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        counter = 0
        # Inicializa arreglo vacio para almacenar las caracteristicas.
        features = np.empty(0, dtype='i')
        # Recorre cada contorno encontrado.
        for cnt in contours:
            # Calcula el area del contorno.
            area = cv2.contourArea(cnt)
            # Verifica si el area del contorno cumple con las condiciones deseadas.
            if 20 < area < 700:
                # Si cumple, aumenta el contador.
                counter += 1
        # Agrega el contador al arreglo de caracteristicas.
        features = np.append(features, [counter])
        # Retorna el arreglo de caracteristicas.
        return features

    # Inicializa arreglos vacios para almacenar las caracteristicas y las etiquetas.
    X = np.empty(0, dtype='i')
    Y = np.empty(0, dtype='i')
    # Recorre cada imagen infectada.
    for image in infected_images:
        # Extrae caracteristicas de la imagen.
        feature_vector = extract_features(image)
        # Agrega caracteristicas al arreglo X.
        X = np.append(X, feature_vector)
        # Agrega etiqueta 1 al arreglo Y (indica que la imagen es infectada).
        Y = np.append(Y, 1)
    # Recorre cada imagen sana.
    for image in healthy_images:
        # Extrae caracteristicas de la imagen.
        feature_vector = extract_features(image)
        # Agrega caracteristicas al arreglo X.
        X = np.append(X, feature_vector)
        # Agrega etiqueta 0 al arreglo Y (indica que la imagen es sana).
        Y = np.append(Y, 0)

    # Normaliza los datos de la matriz X.
    X = X.reshape(-1, 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Divide la matriz X en dos conjuntos de datos, uno para entrenar un modelo de aprendizaje automático y otro para
    # evaluarlo.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # Convierte la matriz Y_train en una matriz 1D. Esto es necesario para que la forma de la matriz Y_train coincida
    # con lo que se espera en el método fit() del modelo de aprendizaje automático.
    Y_train = Y_train.ravel()

    # Entrena un modelo de clasificación basado en árboles de decisión.
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, Y_train)

    # Hacer predicciones en el conjunto de prueba
    Y_pred = model.predict(X_test)

    # Calcular la matriz de confusión
    conf_matrix = confusion_matrix(Y_test, Y_pred)

    # Calcular las métricas
    tp = conf_matrix[0][0]
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]
    tn = conf_matrix[1][1]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Imprimir las métricas
    print('******************************** {} ********************************'.format('Métricas de rendimiento'))
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Specificity: ", specificity)
    print("F1-score: ", f1_score)

    # Datos a graficar
    metrics = ['Precision', 'Recall', 'Specificity', 'F1-score']
    values = [precision, recall, specificity, f1_score]
    # Crea el gráfico de barras
    plt.bar(metrics, values)
    # Personalizar el gráfico
    plt.title('Métricas de rendimiento del modelo')
    plt.xlabel('Métrica')
    plt.ylabel('Valor')
    # Mostrar el gráfico
    # plt.show()

    # Evalúa la precisión del modelo de clasificación entrenado.
    accuracy = model.score(X_test, Y_test)
    print('Precisión del modelo de clasificación entrenado:', accuracy)

    # Utiliza el modelo entrenado para ver si una nueva imagen es infectada o sana.
    def predict(img_path):
        # Leer imagen
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"No se puede leer la imagen en {img_path}")
        # Redimensiona la imágenes a un tamaño de 720x720 píxeles.
        img = cv2.resize(img, (720, 720))
        # Extraer características
        feature_vector = extract_features(img)
        if feature_vector is None:
            raise ValueError("No se pueden extraer características de la imagen")
        # Normalizar características
        feature_vector = scaler.transform([feature_vector])
        # Realizar predicción
        prediction = model.predict(feature_vector)
        print("Predicción: ", prediction[0])
        return prediction

    print('************************* {} *************************'.format('Pruebas'))
    # Test hoja sana
    print('Sanas')
    predict('dataset/test/healthy-leaves/healthy-leaves (1).jpg')
    predict('dataset/test/healthy-leaves/healthy-leaves (2).jpg')
    predict('dataset/test/healthy-leaves/healthy-leaves (3).jpg')
    predict('dataset/test/healthy-leaves/healthy-leaves (4).jpg')
    predict('dataset/test/healthy-leaves/healthy-leaves (5).jpg')
    predict('dataset/test/healthy-leaves/healthy-leaves (6).jpg')
    predict('dataset/test/healthy-leaves/healthy-leaves (7).jpg')
    predict('dataset/test/healthy-leaves/healthy-leaves (8).jpg')
    predict('dataset/test/healthy-leaves/healthy-leaves (9).jpg')
    predict('dataset/test/healthy-leaves/healthy-leaves (10).jpg')

    # Test hoja infectada
    print('Infectada')
    predict('dataset/test/ceramidia-viridis/ceramidia-viridis (1).jpg')
    predict('dataset/test/ceramidia-viridis/ceramidia-viridis (2).jpg')
    predict('dataset/test/ceramidia-viridis/ceramidia-viridis (3).jpg')
    predict('dataset/test/ceramidia-viridis/ceramidia-viridis (4).jpg')
    predict('dataset/test/ceramidia-viridis/ceramidia-viridis (5).jpg')
    predict('dataset/test/ceramidia-viridis/ceramidia-viridis (6).jpg')
    predict('dataset/test/ceramidia-viridis/ceramidia-viridis (7).jpg')
    predict('dataset/test/ceramidia-viridis/ceramidia-viridis (8).jpg')
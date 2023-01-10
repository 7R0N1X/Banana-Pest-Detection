import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    # Carga dos listas de imágenes, una con imágenes infectadas y otra con imágenes sanas.
    creamidia_viridis = [f'dataset/pests/ceramidia-viridis/{img}' for img in os.listdir('dataset/pests/ceramidia-viridis')]
    healthy_leaves = [f'dataset/pests/healthy-leaves/{img}' for img in os.listdir('dataset/pests/healthy-leaves')]

    infected_images = []
    healthy_images = []
    # print('************************* {} *************************'.format('Infected Images'))
    for image_file in creamidia_viridis:
        image = cv2.imread(image_file)
        infected_images.append(image)
        # print(image_file)
    # print('************************* {} *************************'.format('Healthy Images'))
    for image_file in healthy_leaves:
        image = cv2.imread(image_file)
        healthy_images.append(image)
        # print(image_file)
    print('************************* {} *************************'.format('Total de imágenes cargadas'))
    print(f'Imágenes infectadas: {len(infected_images)}')
    print(f'Imágenes sanas: {len(healthy_images)}')

    # Redimensiona las imágenes de plantas infectadas y sanas a un tamaño de 128x128 píxeles.
    for i in range(len(infected_images)):
        infected_images[i] = cv2.resize(infected_images[i], (150, 150))
    for i in range(len(healthy_images)):
        healthy_images[i] = cv2.resize(healthy_images[i], (150, 150))

    # Extrae características de las imágenes de plantas infectadas y sanas.
    def extract_features(img):
        features = np.empty(0, dtype='i')

        white_bajo = np.array([0, 0, 20], np.uint8)
        white_alto = np.array([0, 255, 255], np.uint8)

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask_green = cv2.inRange(img_hsv, white_bajo, white_alto)
        mask_green = cv2.dilate(mask_green, None, iterations=1)
        mask_green = cv2.erode(mask_green, None, iterations=1)

        cnts, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        counter = 0
        for c in cnts:
            area = cv2.contourArea(c)
            if 20 < area < 5000:
                counter += 1
        features = np.append(features, counter)
        return features

    X = np.empty(0, dtype='i')
    Y = np.empty(0, dtype='i')
    for image in infected_images:
        feature_vector = extract_features(image)
        X = np.append(X, feature_vector)
        Y = np.append(Y, 1)  # Indica que la imagen es infectada.
    for image in healthy_images:
        feature_vector = extract_features(image)
        X = np.append(X, feature_vector)
        Y = np.append(Y, 0)  # Indica que la imagen es sana.

    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    # Normaliza los datos de la matriz X.
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

    # Evalúa la precisión del modelo de clasificación entrenado.
    accuracy = model.score(X_test, Y_test)
    print('Precisión:', accuracy)

    # Utiliza el modelo entrenado para ver si una nueva imagen es infectada o sana.
    def predict(img_path):
        # Leer imagen
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"No se puede leer la imagen en {img_path}")

        # Extraer características
        feature_vector = extract_features(img)
        if feature_vector is None:
            raise ValueError("No se pueden extraer características de la imagen")

        # Normalizar características
        feature_vector = scaler.transform([feature_vector])

        # Realizar predicción
        prediction = model.predict(feature_vector)
        print("Predicción: ", prediction)
        return prediction

    # Test hoja sana
    predict('dataset/pests/healthy-leaves/healthy-leaves (52).jpg')
    # Test hoja infectada
    predict('dataset/pests/ceramidia-viridis/ceramidia-viridis (10).jpg')
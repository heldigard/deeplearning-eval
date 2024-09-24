# Deeplearning-Eval

**Integrantes**
* Eldigardo Camacho

# Image Classification

En los notebooks de Image Classification se va a realizar la clasificación de imágenes utilizando una red neuronal convolucional. Se va a utilizar la base de datos de **Natural Scene**, Estos son datos de imágenes de escenas naturales de todo el mundo. Las imágenes se han recopilado de la web y etiquetado manualmente. Las imágenes están divididas en 6 categorías. Estas son las categorías:

1. Buildings
2. Forest
3. Glacier
4. Mountain
5. Sea
6. Street

- En el notebook **Image_Classification_CNN.ipynb** se utiliza una red neuronal entrenada manualmente.
- En el notebook **Image_Classification_VGG16_ResNet50_InceptionV3.ipynb** se utiliza una red neuronal preentrenada con el conjunto de datos **ImageNet**. Estas redes neuronales preentrenadas se utilizan para la extracción de características de las imágenes y se utiliza un clasificador lineal para la clasificación de las imágenes. Las redes neuronales preentrenadas utilizadas son **VGG16**, **ResNet50** e **InceptionV3**. Estas redes neuronales preentrenadas se utilizan para la extracción de características de las imágenes y se utiliza un clasificador lineal para la clasificación de las imágenes. Las características extraídas se utilizan como entrada para un clasificador lineal, que se entrena con los datos de entrenamiento. El clasificador lineal se utiliza para predecir la clase de las imágenes de prueba.

# YOLO deteccion en tiempo real

En el archivo Camera_YOLO.py se utiliza el modelo YOLO v3, para detectar objetos del flujo de video de la camara.


'''
Paso 1: Preparación del Entorno
1. Asegúrate de tener Python y OpenCV instalados.
2. Descarga o selecciona una imagen binarias y en niveles de gris para realizar las
operaciones.
Paso 2: Carga y Visualización de la Imagen
1. Carga la imagen binaria o en niveles de gris utilizando OpenCV.
2. Visualiza la imagen original.
A. Código para manipular imágenes binarias: este código en python, modifique o
complemente si es necesario.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
# Cargar la imagen en escala de grises
imagen = cv2.imread('imagen_binaria.jpg', 0)
# Mostrar la imagen original
plt.imshow(imagen, cmap='gray')
plt.title('Imagen Original')
plt.show()

'''

Paso 3: Operación de Erosión
1. Aplica la operación de erosión a la imagen.
2. Visualiza el resultado de la erosión.
Código: este código en python, modifique o complemente si es necesario.
'''

# Definir el kernel para la erosión
kernel = np.ones((5,5), np.uint8)
# Aplicar la operación de erosión
imagen_erosionada = cv2.erode(imagen, kernel, iterations = 1)
# Mostrar la imagen erosionada
plt.imshow(imagen_erosionada, cmap='gray')
plt.title('Imagen Erosionada')
plt.show()

'''
Paso 4: Operación de Dilatación
1. Aplica la operación de dilatación a la imagen.
2. Visualiza el resultado de la dilatación.
'''

# Aplicar la operación de dilatación
imagen_dilatada = cv2.dilate(imagen, kernel, iterations = 1)
# Mostrar la imagen dilatada
plt.imshow(imagen_dilatada, cmap='gray')
plt.title('Imagen Dilatada')
plt.show()

'''
Paso 5: Operación de Apertura
1. Aplica la operación de apertura a la imagen de dos formas;
     Método tradicional: genera las líneas de código necesarios empleando los
    operadores básicos de la dilatación y erosión.
     Usa la función cv2.MORPH_OPEN
2. Visualiza el resultado de la apertura.
'''
# Aplicar la operación de apertura usando los operadores básicos
# de la dilatación y erosión, genera el código necesario
imagen_aperturaTradicional = ...
# Mostrar la imagen con apertura
plt.imshow(imagen_aperturaTradicional cmap='gray')
plt.title('Imagen con Apertura del modo base')
plt.show()
# Aplicar la operación de apertura usando la función definida para ello en Open CV
imagen_apertura = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
# Mostrar la imagen con apertura
plt.imshow(imagen_apertura, cmap='gray')
plt.title('Imagen con Apertura')
plt.show()


'''
Paso 6: Operación de Cierre
1. Aplica la operación de cierre a la imagen.
     Método tradicional: genera las líneas de código necesarios empleando los
    operadores básicos de la dilatación y erosión.
     Usa la función cv2.MORPH_CLOSE
2. Visualiza el resultado del cierre.
'''

# Aplicar la operación del cierre usando los operadores básicos
# de la dilatación y erosión, genera el código necesario
imagen_cierreTradicional = ...
# Mostrar la imagen con el cierre
plt.imshow(imagen_cierreTradicional cmap='gray')
plt.title('Imagen con el Cierre del modo base')
plt.show()
# Aplicar la operación de cierre
imagen_cierre = cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)
# Mostrar la imagen con cierre
plt.imshow(imagen_cierre, cmap='gray')
plt.title('Imagen con Cierre')
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen binaria
imagen = cv2.imread('img_bin1.jpg', 0)

# Mostrar la imagen original
plt.imshow(imagen, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')
plt.show()

# Definir el kernel para las operaciones morfológicas
kernel = np.ones((5, 5), np.uint8)

# --- Erosión ---
imagen_erosionada = cv2.erode(imagen, kernel, iterations=1)
plt.imshow(imagen_erosionada, cmap='gray')
plt.title('Imagen Erosionada')
plt.axis('off')
plt.show()

# --- Dilatación ---
imagen_dilatada = cv2.dilate(imagen, kernel, iterations=1)
plt.imshow(imagen_dilatada, cmap='gray')
plt.title('Imagen Dilatada')
plt.axis('off')
plt.show()

# --- Apertura Tradicional: erosión seguida de dilatación ---
imagen_apertura_trad = cv2.dilate(cv2.erode(imagen, kernel, iterations=1), kernel, iterations=1)
plt.imshow(imagen_apertura_trad, cmap='gray')
plt.title('Apertura (Tradicional)')
plt.axis('off')
plt.show()

# --- Apertura con función ---
imagen_apertura = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
plt.imshow(imagen_apertura, cmap='gray')
plt.title('Apertura (cv2.MORPH_OPEN)')
plt.axis('off')
plt.show()

# --- Cierre Tradicional: dilatación seguida de erosión ---
imagen_cierre_trad = cv2.erode(cv2.dilate(imagen, kernel, iterations=1), kernel, iterations=1)
plt.imshow(imagen_cierre_trad, cmap='gray')
plt.title('Cierre (Tradicional)')
plt.axis('off')
plt.show()

# --- Cierre con función ---
imagen_cierre = cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)
plt.imshow(imagen_cierre, cmap='gray')
plt.title('Cierre (cv2.MORPH_CLOSE)')
plt.axis('off')
plt.show()

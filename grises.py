import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
imagen = cv2.imread('img_gs3.jpg', cv2.IMREAD_GRAYSCALE)

# Mostrar la imagen original
plt.imshow(imagen, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')
plt.show()

# Definir el kernel para las operaciones
kernel = np.ones((5, 5), np.uint8)

# --- Erosión ---
imagen_erosionada = cv2.erode(imagen, kernel, iterations=1)
plt.imshow(imagen_erosionada, cmap='gray')
plt.title('Erosión')
plt.axis('off')
plt.show()

# --- Dilatación ---
imagen_dilatada = cv2.dilate(imagen, kernel, iterations=1)
plt.imshow(imagen_dilatada, cmap='gray')
plt.title('Dilatación')
plt.axis('off')
plt.show()

# --- Apertura Tradicional ---
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

# --- Cierre Tradicional ---
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

'''
Morfología en Laticces: 
- Gradiente morfológico (simétrico, por erosión y por dilatación)
- Transformada Bot y Top Hat, 
- Filtros para suavizado.
'''

# Gradiente Simétrico
grad_sim = cv2.subtract(cv2.dilate(imagen, kernel), cv2.erode(imagen, kernel))
plt.imshow(grad_sim, cmap='gray')
plt.title('Gradiente Simétrico')
plt.axis('off')
plt.show()

# Gradiente por erosión (bordes internos)
grad_erosion = cv2.subtract(imagen, cv2.erode(imagen, kernel))
plt.imshow(grad_erosion, cmap='gray')
plt.title('Gradiente por Erosión')
plt.axis('off')
plt.show()

# Gradiente por dilatación (bordes externos)
grad_dilat = cv2.subtract(cv2.dilate(imagen, kernel), imagen)
plt.imshow(grad_dilat, cmap='gray')
plt.title('Gradiente por Dilatación')
plt.axis('off')
plt.show()

# Top Hat: original - apertura (resalta detalles brillantes pequeños)
top_hat = cv2.morphologyEx(imagen, cv2.MORPH_TOPHAT, kernel)
plt.imshow(top_hat, cmap='gray')
plt.title('Top Hat')
plt.axis('off')
plt.show()

# Black Hat: cierre - original (resalta detalles oscuros pequeños)
black_hat = cv2.morphologyEx(imagen, cv2.MORPH_BLACKHAT, kernel)
plt.imshow(black_hat, cmap='gray')
plt.title('Black Hat')
plt.axis('off')
plt.show()

# Suavizado morfológico: apertura seguida de cierre
suavizado = cv2.morphologyEx(cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel),
                             cv2.MORPH_CLOSE, kernel)
plt.imshow(suavizado, cmap='gray')
plt.title('Suavizado Morfológico')
plt.axis('off')
plt.show()

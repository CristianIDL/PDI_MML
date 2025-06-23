import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen binaria
imagen = cv2.imread('img_bin4.jpg', 0)

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

'''
Morfología Binaria: 
- Frontera, 
- Adelgazamiento, 
- Transformada Hit or Miss, 
- Esqueleto Morfológico.
'''

# Frontera = Imagen original - erosión
frontera = cv2.subtract(imagen, cv2.erode(imagen, kernel, iterations=1))
plt.imshow(frontera, cmap='gray')
plt.title('Frontera Morfológica')
plt.axis('off')
plt.show()

# Definir dos kernels (EE1 para objeto, EE2 para fondo)
EE1 = np.array([[0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]], dtype=np.uint8)

EE2 = np.array([[1, 0, 1],
                [0, 0, 0],
                [1, 0, 1]], dtype=np.uint8)

# Convertir imagen a binaria (por si no está estrictamente binaria)
_, imagen_bin = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)

# Complemento de la imagen
imagen_complemento = cv2.bitwise_not(imagen_bin)

# Hit-or-Miss = Erosión(imagen, EE1) ∩ Erosión(imagen_complemento, EE2)
erosion_objeto = cv2.erode(imagen_bin, EE1)
erosion_fondo = cv2.erode(imagen_complemento, EE2)
hit_or_miss = cv2.bitwise_and(erosion_objeto, erosion_fondo)

plt.imshow(hit_or_miss, cmap='gray')
plt.title('Transformada Hit-or-Miss')
plt.axis('off')
plt.show()

# Crear esqueleto vacío
esqueleto = np.zeros(imagen.shape, np.uint8)
elemento = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
img_temp = imagen.copy()

while True:
    apertura = cv2.morphologyEx(img_temp, cv2.MORPH_OPEN, elemento)
    temp = cv2.subtract(img_temp, apertura)
    esqueleto = cv2.bitwise_or(esqueleto, temp)
    img_temp = cv2.erode(img_temp, elemento)
    
    if cv2.countNonZero(img_temp) == 0:
        break

plt.imshow(esqueleto, cmap='gray')
plt.title('Esqueleto Morfológico')
plt.axis('off')
plt.show()

'''
Morfología en Laticces: 
- Gradiente morfológico (simétrico, por erosión y por dilatación)
- Transformada Bot y Top Hat, 
- Filtros para suavizado.
'''

# Gradiente Simétrico
grad_sim = cv2.subtract(cv2.dilate(imagen, kernel, iterations=1),
                        cv2.erode(imagen, kernel, iterations=1))
plt.imshow(grad_sim, cmap='gray')
plt.title('Gradiente Morfológico Simétrico')
plt.axis('off')
plt.show()

# Gradiente por erosión
grad_erosion = cv2.subtract(imagen, cv2.erode(imagen, kernel, iterations=1))
plt.imshow(grad_erosion, cmap='gray')
plt.title('Gradiente por Erosión')
plt.axis('off')
plt.show()

# Gradiente por dilatación
grad_dilat = cv2.subtract(cv2.dilate(imagen, kernel, iterations=1), imagen)
plt.imshow(grad_dilat, cmap='gray')
plt.title('Gradiente por Dilatación')
plt.axis('off')
plt.show()

# Top Hat
top_hat = cv2.morphologyEx(imagen, cv2.MORPH_TOPHAT, kernel)
plt.imshow(top_hat, cmap='gray')
plt.title('Top Hat')
plt.axis('off')
plt.show()

# Black Hat
black_hat = cv2.morphologyEx(imagen, cv2.MORPH_BLACKHAT, kernel)
plt.imshow(black_hat, cmap='gray')
plt.title('Black Hat')
plt.axis('off')
plt.show()

# Suavizado: Apertura seguida de cierre
suavizado = cv2.morphologyEx(cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel),
                             cv2.MORPH_CLOSE, kernel)
plt.imshow(suavizado, cmap='gray')
plt.title('Suavizado Morfológico')
plt.axis('off')
plt.show()

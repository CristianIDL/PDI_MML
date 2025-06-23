import cv2
import numpy as np
import matplotlib.pyplot as plt

# Kernel por defecto
DEFAULT_KERNEL = np.ones((5, 5), np.uint8)

# --- Operaciones básicas ---
def erosion(imagen, kernel=DEFAULT_KERNEL):
    return cv2.erode(imagen, kernel, iterations=1)

def dilatacion(imagen, kernel=DEFAULT_KERNEL):
    return cv2.dilate(imagen, kernel, iterations=1)

def apertura(imagen, kernel=DEFAULT_KERNEL):
    return cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)

def apertura_tradicional(imagen, kernel=DEFAULT_KERNEL):
    return dilatacion(erosion(imagen, kernel), kernel)

def cierre(imagen, kernel=DEFAULT_KERNEL):
    return cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)

def cierre_tradicional(imagen, kernel=DEFAULT_KERNEL):
    return erosion(dilatacion(imagen, kernel), kernel)

# --- Operaciones avanzadas ---

'''
Morfología Binaria: 
- Frontera, 
- Adelgazamiento, 
- Transformada Hit or Miss, 
- Esqueleto Morfológico.
'''

def frontera(imagen, kernel=DEFAULT_KERNEL):
    return cv2.subtract(imagen, erosion(imagen, kernel))

def hit_or_miss(imagen_binaria, ee1, ee2):
    complemento = cv2.bitwise_not(imagen_binaria)
    eros1 = cv2.erode(imagen_binaria, ee1)
    eros2 = cv2.erode(complemento, ee2)
    return cv2.bitwise_and(eros1, eros2)

def esqueleto(imagen_binaria, kernel=None):
    if kernel is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    esq = np.zeros(imagen_binaria.shape, np.uint8)
    temp = imagen_binaria.copy()
    
    while True:
        abierto = apertura(temp, kernel)
        resta = cv2.subtract(temp, abierto)
        esq = cv2.bitwise_or(esq, resta)
        temp = erosion(temp, kernel)
        if cv2.countNonZero(temp) == 0:
            break
    return esq

'''
Morfología en Laticces: 
- Gradiente morfológico (simétrico, por erosión y por dilatación)
- Transformada Bot y Top Hat, 
- Filtros para suavizado.
'''

def gradiente_simetrico(imagen, kernel=DEFAULT_KERNEL):
    return cv2.subtract(dilatacion(imagen, kernel), erosion(imagen, kernel))

def gradiente_erosion(imagen, kernel=DEFAULT_KERNEL):
    return cv2.subtract(imagen, erosion(imagen, kernel))

def gradiente_dilatacion(imagen, kernel=DEFAULT_KERNEL):
    return cv2.subtract(dilatacion(imagen, kernel), imagen)

def mostrarIMG(imagen,titulo):
    plt.imshow(imagen, cmap='gray')
    plt.title(titulo)
    plt.axis('off')
    plt.show()

def top_hat(imagen, kernel=DEFAULT_KERNEL):
    return cv2.morphologyEx(imagen, cv2.MORPH_TOPHAT, kernel)

def black_hat(imagen, kernel=DEFAULT_KERNEL):
    return cv2.morphologyEx(imagen, cv2.MORPH_BLACKHAT, kernel)

def suavizado_morfologico(imagen, kernel=DEFAULT_KERNEL):
    return cierre(apertura(imagen, kernel), kernel)

def mostrarComparacion(original, procesada, tit1='Original', tit2='Procesada'):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title(tit1)
    axs[0].axis('off')
    axs[1].imshow(procesada, cmap='gray')
    axs[1].set_title(tit2)
    axs[1].axis('off')
    plt.show()

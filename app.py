import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

from morfologia import *

# --- Funciones de GUI ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesamiento Morfológico")
        self.root.geometry("900x500")

        self.panel_original = tk.Label(root)
        self.panel_procesada = tk.Label(root)
        self.imagen = None

        # Botón de cargar imagen
        btn_cargar = tk.Button(root, text="Cargar Imagen", command=self.cargar_imagen)
        btn_cargar.pack(pady=10)

        # Menú desplegable de operaciones
        self.opciones = [
            "Erosión", "Dilatación", "Apertura", "Cierre",
            "Frontera", "Hit or Miss", "Gradiente Simétrico",
            "Gradiente por Erosión", "Gradiente por Dilatación",
            "Top Hat", "Black Hat", "Suavizado"
        ]
        self.var_op = tk.StringVar()
        self.var_op.set(self.opciones[0])
        combo = ttk.Combobox(root, textvariable=self.var_op, values=self.opciones, state="readonly")
        combo.pack()

        # Botón de aplicar operación
        btn_aplicar = tk.Button(root, text="Aplicar Operación", command=self.aplicar_operacion)
        btn_aplicar.pack(pady=10)

        # Marcos para mostrar imágenes
        self.panel_original.pack(side="left", padx=10)
        self.panel_procesada.pack(side="right", padx=10)

    def cargar_imagen(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.imagen = cv2.imread(file_path, 0)
            self.mostrar_imagen(self.imagen, self.panel_original, "Original")

    def aplicar_operacion(self):
        if self.imagen is None:
            return

        operacion = self.var_op.get()
        resultado = None

        if operacion == "Erosión":
            resultado = erosion(self.imagen)
        elif operacion == "Dilatación":
            resultado = dilatacion(self.imagen)
        elif operacion == "Apertura":
            resultado = apertura(self.imagen)
        elif operacion == "Cierre":
            resultado = cierre(self.imagen)
        elif operacion == "Frontera":
            resultado = frontera(self.imagen)
        elif operacion == "Hit or Miss":
            # Convertir imagen a binaria (umbral simple)
            _, imagen_bin = cv2.threshold(self.imagen, 127, 255, cv2.THRESH_BINARY)

            # Definir elementos estructurantes para objeto y fondo
            EE1 = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=np.uint8)

            EE2 = np.array([[1, 0, 1],
                            [0, 0, 0],
                            [1, 0, 1]], dtype=np.uint8)

            resultado = hit_or_miss(imagen_bin, EE1, EE2)
        elif operacion == "Gradiente Simétrico":
            resultado = gradiente_simetrico(self.imagen)
        elif operacion == "Gradiente por Erosión":
            resultado = gradiente_erosion(self.imagen)
        elif operacion == "Gradiente por Dilatación":
            resultado = gradiente_dilatacion(self.imagen)
        elif operacion == "Top Hat":
            resultado = top_hat(self.imagen)
        elif operacion == "Black Hat":
            resultado = black_hat(self.imagen)
        elif operacion == "Suavizado":
            resultado = suavizado_morfologico(self.imagen)

        if resultado is not None:
            self.mostrar_imagen(resultado, self.panel_procesada, "Procesada")

    def mostrar_imagen(self, img_cv, panel, titulo):
        # Convertir a imagen para Tkinter
        img_pil = Image.fromarray(img_cv)
        img_pil = img_pil.resize((400, 400))
        img_tk = ImageTk.PhotoImage(img_pil)
        panel.configure(image=img_tk, text=titulo, compound="top")
        panel.image = img_tk


# --- Ejecutar aplicación ---
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

import tkinter as tk
from tkinter import Canvas, Button
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageGrab, ImageOps
import matplotlib.pyplot as plt

# Загрузка обученной модели
model = load_model("mnist_model_v2.h5")
image_size = (28, 28)


def preprocess_image(image):
    """Просто инвертирует изображение и приводит его к размеру 28x28 без обрезки."""
    image = image.convert("L")  # Перевод в Ч/Б
    image = ImageOps.invert(image)  # Инвертируем цвета (фон черный, цифра белая)

    # Масштабируем изображение к 28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    plt.imshow(image, cmap="gray")
    plt.title("Обработанное изображение")
    plt.show()

    # Преобразуем в массив и нормализуем
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание цифр")

        self.canvas = Canvas(root, width=300, height=300, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        self.predict_button = Button(root, text="Распознать", command=self.recognize_digit)
        self.predict_button.pack()

        self.clear_button = Button(root, text="Очистить", command=self.clear_canvas)
        self.clear_button.pack()

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="black", outline="black")

    def get_canvas_image(self):
        """Захватывает изображение с Canvas и возвращает его как объект PIL Image."""
        self.canvas.update()  # Обновляем Canvas перед захватом
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        image = ImageGrab.grab(bbox=(x, y, x1, y1))  # Захват всей области Canvas
        return image

    def predict_digit(self):
        """Получает изображение с Canvas, обрабатывает и передает в модель."""
        image = self.get_canvas_image()  # Теперь `self` доступен
        image = preprocess_image(image)
        prediction = model.predict(image)

        # Отладка: Вывод сырых предсказаний
        print(f"Сырые предсказания: {prediction}")

        return np.argmax(prediction)

    def recognize_digit(self):
        digit = self.predict_digit()
        print(f"Распознанная цифра: {digit}")

    def clear_canvas(self):
        self.canvas.delete("all")


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()

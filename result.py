import tkinter as tk
from tkinter import Canvas, Button
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageGrab, ImageOps, ImageEnhance
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu  # Для адаптивной бинаризации

# Загрузка обученной модели
model = load_model("mnist_model_v2.h5")
image_size = (28, 28)

def center_and_crop(image, padding=20):
    """Центрирует и обрезает изображение вокруг нарисованной области с увеличенным запасом."""
    image_array = np.array(image)
    # Найти ненулевые пиксели (после инверсии ненулевые — это белые пиксели)
    nonzero = np.where(image_array < 255)
    if len(nonzero[0]) == 0:  # Если ничего не нарисовано
        return image

    top, bottom = np.min(nonzero[0]), np.max(nonzero[0])
    left, right = np.min(nonzero[1]), np.max(nonzero[1])

    # Увеличенный запас
    top = max(0, top - padding)
    bottom = min(image_array.shape[0], bottom + padding)
    left = max(0, left - padding)
    right = min(image_array.shape[1], right + padding)

    # Обрезать вокруг нарисованной области
    image = image.crop((left, top, right, bottom))
    return image

def preprocess_image(image):
    """Обрабатывает изображение, обрезает его, масштабирует и нормализует."""
    # Отладка: Показываем захваченное изображение до обработки
    plt.imshow(image)
    plt.title("Захваченное изображение (до обработки)")
    plt.show()

    # Обрезаем центральную область (например, 260x260 из 280x280)
    canvas_size = 280
    drawing_area = 260
    left = (canvas_size - drawing_area) // 2
    top = (canvas_size - drawing_area) // 2
    right = left + drawing_area
    bottom = top + drawing_area
    image = image.crop((left, top, right, bottom))

    # Преобразуем в Ч/Б
    image = image.convert("L")

    # Улучшаем контраст
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)  # Увеличиваем контраст в 2 раза

    # Адаптивная бинаризация
    image_array = np.array(image)
    thresh = threshold_otsu(image_array)
    image_array = (image_array > thresh) * 255
    image = Image.fromarray(image_array.astype(np.uint8))

    # Инвертируем цвета (фон чёрный, цифра белая)
    image = ImageOps.invert(image)

    # Центрируем и обрезаем вокруг нарисованной области
    image = center_and_crop(image, padding=30)  # Увеличиваем запас

    # Масштабируем с промежуточным размером
    image = image.resize((56, 56), Image.Resampling.LANCZOS)  # Промежуточный размер
    image = image.resize((28, 28), Image.Resampling.LANCZOS)  # Финальный размер

    # Отображаем обработанное изображение для отладки
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

        # Canvas размером 280x280
        self.canvas_size = 280
        self.canvas = Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        # Добавляем визуальную рамку для области рисования
        self.canvas.create_rectangle(10, 10, self.canvas_size-10, self.canvas_size-10, outline="gray", dash=(2, 2))

        self.predict_button = Button(root, text="Распознать", command=self.recognize_digit)
        self.predict_button.pack()

        self.clear_button = Button(root, text="Очистить", command=self.clear_canvas)
        self.clear_button.pack()

    def draw(self, event):
        x, y = event.x, event.y
        # Увеличиваем толщину линии для лучшей видимости после масштабирования
        self.canvas.create_oval(x - 15, y - 15, x + 15, y + 15, fill="black", outline="black")

    def get_canvas_image(self):
        """Захватывает изображение с Canvas и возвращает его как объект PIL Image."""
        self.canvas.update()  # Обновляем Canvas перед захватом
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        # Добавляем небольшой отступ, чтобы избежать захвата рамок окна
        offset = 2
        image = ImageGrab.grab(bbox=(x + offset, y + offset, x1 - offset, y1 - offset))

        # Отладка: Показываем сырое захваченное изображение
        plt.imshow(image)
        plt.title("Захваченное изображение (сырое)")
        plt.show()

        return image

    def predict_digit(self):
        """Получает изображение с Canvas, обрабатывает и передает в модель."""
        image = self.get_canvas_image()
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
        # Восстанавливаем рамку после очистки
        self.canvas.create_rectangle(10, 10, self.canvas_size-10, self.canvas_size-10, outline="gray", dash=(2, 2))


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
import os
import tkinter as tk
from tkinter import simpledialog, filedialog
from PIL import Image, ImageDraw, ImageOps

# Создаём папку для датасета
dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)


class DrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Рисование чисел")

        self.canvas = tk.Canvas(root, width=300, height=300, bg="white")
        self.canvas.pack()

        self.btn_save = tk.Button(root, text="Сохранить", command=self.save_image)
        self.btn_save.pack()

        self.btn_clear = tk.Button(root, text="Очистить", command=self.clear_canvas)
        self.btn_clear.pack()

        self.btn_eraser = tk.Button(root, text="Ластик", command=self.toggle_eraser)
        self.btn_eraser.pack()

        self.btn_load = tk.Button(root, text="Загрузить изображение", command=self.load_image)
        self.btn_load.pack()

        self.eraser_mode = False  # Флаг для режима ластика
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = Image.new("L", (300, 300), 255)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x, y = event.x, event.y
        color = 255 if self.eraser_mode else 0  # Белый цвет для стирания
        self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="white" if self.eraser_mode else "black",
                                outline="white" if self.eraser_mode else "black")
        self.draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color)

    def toggle_eraser(self):
        self.eraser_mode = not self.eraser_mode  # Переключение режима ластика

    def save_image(self):
        number = simpledialog.askstring("Сохранение", "Введите число (1-100):")
        if number and number.isdigit():
            number = int(number)
            if 1 <= number <= 100:
                folder = os.path.join(dataset_path, str(number))
                os.makedirs(folder, exist_ok=True)
                count = len(os.listdir(folder))
                filename = os.path.join(folder, f"{count + 1}.png")
                self.image.resize((64, 64)).save(filename)
                print(f"Сохранено: {filename}")
                self.clear_canvas()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (300, 300), 255)
        self.draw = ImageDraw.Draw(self.image)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")])
        if not file_path:
            return

        image = Image.open(file_path).convert("L")  # Открываем в черно-белом режиме
        image = ImageOps.invert(image)  # Инвертируем цвета, если нужно
        image = image.resize((64, 64))  # Меняем размер

        number = simpledialog.askstring("Сохранение", "Введите число (1-100):")
        if number and number.isdigit():
            number = int(number)
            if 1 <= number <= 100:
                folder = os.path.join(dataset_path, str(number))
                os.makedirs(folder, exist_ok=True)
                count = len(os.listdir(folder))
                filename = os.path.join(folder, f"{count + 1}.png")
                image.save(filename)
                print(f"Изображение сохранено в {filename}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()

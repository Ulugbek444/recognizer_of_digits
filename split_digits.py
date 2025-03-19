import os
import cv2
import numpy as np
from PIL import Image

dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)
source_path = "dataset"  # Папка с изображениями чисел


def split_number_image(image_path, number_label):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # Сортируем слева направо

    digits = list(str(number_label))  # Разбиваем число на отдельные цифры

    if len(contours) != len(digits):
        print(f"⚠ Предупреждение: найдено {len(contours)} цифр, но число состоит из {len(digits)} цифр!")

    for i, (contour, digit_label) in enumerate(zip(contours, digits)):
        x, y, w, h = cv2.boundingRect(contour)
        digit = image[y:y + h, x:x + w]  # Вырезаем цифру
        digit = cv2.resize(digit, (64, 64))  # Меняем размер
        digit = Image.fromarray(digit)

        folder = os.path.join(dataset_path, digit_label)
        os.makedirs(folder, exist_ok=True)
        count = len(os.listdir(folder))
        filename = os.path.join(folder, f"{count + 1}.png")
        digit.save(filename)
        print(f"Сохранено: {filename}")


if __name__ == "__main__":
    if os.path.exists(source_path):
        for folder_name in sorted(os.listdir(source_path)):
            folder_path = os.path.join(source_path, folder_name)
            if os.path.isdir(folder_path) and folder_name.isdigit() and int(folder_name) >= 10:
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.isfile(file_path) and file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        print(f"Обрабатываю {file_name} из {folder_name}...")
                        split_number_image(file_path, folder_name)
    else:
        print(f"Папка {source_path} не найдена.")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Загружаем MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Нормализация данных (делаем значения от 0 до 1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Добавляем канал (для сверточной сети)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Создаем модель CNN
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучаем модель (10 эпох)
model.fit(X_train, y_train, epochs=18, validation_data=(X_test, y_test))
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

# Выводим итоговую точность
print(f"\nИтоговая точность модели: {test_acc:.4f}")
# Сохраняем модель
model.save("mnist_model_v2.h5")
print("Модель обучена и сохранена!")

// Получаем Canvas и контекст
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;

// Настройки рисования
ctx.lineWidth = 15; // Увеличиваем толщину линии
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height); // Белый фон

// Переменная для модели
let model;

// Загружаем модель при старте
async function loadModel() {
    model = await tf.loadLayersModel('/static/model/model.json');
    console.log('Model loaded');
}

// События для рисования
canvas.addEventListener('mousedown', () => drawing = true);
canvas.addEventListener('mouseup', () => drawing = false);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('touchstart', (e) => {
    drawing = true;
    e.preventDefault();
});
canvas.addEventListener('touchend', () => drawing = false);
canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    draw({ offsetX: x, offsetY: y });
});

function draw(e) {
    if (!drawing) return;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    ctx.closePath();
}

function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('prediction').innerText = 'Prediction: None';
}

// Функция для предобработки изображения
function preprocessImage() {
    // Получаем данные изображения с Canvas
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    // Преобразуем в оттенки серого и инвертируем цвета
    const gray = new Float32Array(canvas.width * canvas.height);
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b; // Формула для серого
        gray[i / 4] = 1 - (grayscale / 255); // Инверсия: белый -> 0, чёрный -> 1
    }

    // Масштабируем до 28x28
    const resized = new Float32Array(28 * 28);
    const scale = canvas.width / 28;
    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            let sum = 0;
            let count = 0;
            for (let dy = 0; dy < scale; dy++) {
                for (let dx = 0; dx < scale; dx++) {
                    const px = Math.floor(x * scale + dx);
                    const py = Math.floor(y * scale + dy);
                    if (px < canvas.width && py < canvas.height) {
                        sum += gray[py * canvas.width + px];
                        count++;
                    }
                }
            }
            resized[y * 28 + x] = sum / count;
        }
    }

    // Преобразуем в тензор для TensorFlow.js
    const tensor = tf.tensor4d(resized, [1, 28, 28, 1]);
    return tensor;
}

async function predictDigit() {
    if (!model) {
        alert('Model is still loading, please wait...');
        return;
    }

    // Предобработка изображения
    const tensor = preprocessImage();

    // Предсказание
    const prediction = model.predict(tensor);
    const digit = prediction.argMax(-1).dataSync()[0];
    document.getElementById('prediction').innerText = `Prediction: ${digit}`;

    // Очистка тензора
    tensor.dispose();
    prediction.dispose();
}

// Загружаем модель при загрузке страницы
loadModel();
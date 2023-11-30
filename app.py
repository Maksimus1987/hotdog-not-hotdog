# Разработчик №1 (@Maksimus1987 mask-13@mail.ru) - Импорт необходимых библиотек и инициализация модели классификации изображений
import streamlit as st
from transformers import pipeline 

from PIL import Image
pipeline = pipeline(task="image-classification", model="julien-c/hotdog-not-hotdog")

# Разработчик №2 - Создание веб-страницы с помощью Streamlit и загрузка изображения пользователем
st.title("Hot Dog? Or Not?")
file_name = st.file_uploader("Upload a hot dog candidate image") 

# Разработчик №3 @aleksrf1 aleksrf@gmail.com - Обработка загруженного изображения и получение прогноза модели
if file_name is not None:
    col1, col2 = st.columns(2)
    image = Image.open(file_name)
    col1.image(image, use_column_width=True)
    predictions = pipeline(image)
    col2.header("Probabilities")
# Разработчик №4 (@xbtart xbtart@yandex.ru - Миронов Артур Викторович) - Отображение изображения и прогноза на веб-странице
    for p in predictions:
        col2.subheader(f"{p['label']}: {round(p['score'] * 100, 1)}%")


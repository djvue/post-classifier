# HW3: docker project

## Описание модели и этапов сборки

Используется dataset https://www.kaggle.com/code/josuanicolas/sentiment-excercise/input
Он представляет собой посты твиттера с категорией и оценкой.
Используем его для создания сервиса оценки текстов.

Для препроцессинга используем sklearn.feature_extraction.text и BeautifulSoup.

В качестве модели sklearn PassiveAggressiveClassifier.

Результаты тестирования модели:
```
Acc: 95.1%
confusion matrix:
[[270   8   4]
 [  3 265   5]
 [  2   7 255]]
```

Обученная модель и векторизатор сохраняются библиотекой joblib для последующей загрузки
в приложении веб-сервера на fastapi

## Веб-сервер fastapi

Имеет 1 полезных ендпоинт для оценки текста
```http
POST /api/sentiment
```
```json
{
    "text": "What a good day"
}
```
возвращает
```json
{
    "status": "ok",
    "sentiment": "Positive"
}
```

Текст пропускается через векторизатор и передается модели для получения предсказания.

## Сборка и разворачивание в локальной среде

API веб сервера будет доступно по адресу http://localhost:8000/.

Можно использовать любой из 3 вариантов ниже.

### Python

Установка зависимостей
```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Обучение модели
```sh
python train_model.py
```

Запуск проекта
```sh
uvicorn app:app --reload
```

### Docker

В Dockerfile происходит установка всех зависимостей и обучение модели.

```sh
docker build -t mlops_hw3 .
docker run --rm mlops_hw3
```

### Docker Compose

```sh
docker-compose up -d
```

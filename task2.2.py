import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# Данные пользователей 
data = [
    {"user_id": "u01", "age": 22, "gender": "female", "education_level": "bachelor",
     "messages_per_day": 25, "avg_message_length": 70, "positive_ratio": 0.8,
     "is_active": 1},
    {"user_id": "u02", "age": 30, "gender": "male", "education_level": "master",
     "messages_per_day": 10, "avg_message_length": 50, "positive_ratio": 0.6,
     "is_active": 0},
    {"user_id": "u03", "age": 26, "gender": "female", "education_level": "high_school",
     "messages_per_day": 40, "avg_message_length": 120, "positive_ratio": 0.9,
     "is_active": 1},
    {"user_id": "u04", "age": 35, "gender": "male", "education_level": "bachelor",
     "messages_per_day": 5, "avg_message_length": 20, "positive_ratio": 0.3,
     "is_active": 0},
    {"user_id": "u05", "age": 40, "gender": "female", "education_level": "master",
     "messages_per_day": 60, "avg_message_length": 150, "positive_ratio": 0.7,
     "is_active": 1},
    {"user_id": "u06", "age": 28, "gender": "male", "education_level": "high_school",
     "messages_per_day": 15, "avg_message_length": 80, "positive_ratio": 0.5,
     "is_active": 0},
    {"user_id": "u07", "age": 24, "gender": "female", "education_level": "bachelor",
     "messages_per_day": 30, "avg_message_length": 100, "positive_ratio": 0.85,
     "is_active": 1},
    {"user_id": "u08", "age": 32, "gender": "male", "education_level": "bachelor",
     "messages_per_day": 8, "avg_message_length": 30, "positive_ratio": 0.4,
     "is_active": 0},
    {"user_id": "u09", "age": 29, "gender": "female", "education_level": "master",
     "messages_per_day": 50, "avg_message_length": 130, "positive_ratio": 0.95,
     "is_active": 1},
    {"user_id": "u10", "age": 38, "gender": "male", "education_level": "high_school",
     "messages_per_day": 2, "avg_message_length": 10, "positive_ratio": 0.2,
     "is_active": 0}
]

# Преобразуем в DataFrame
df = pd.DataFrame(data)

# Признаки и целевая переменная
X = df[['age', 'gender', 'education_level', 'messages_per_day', 'avg_message_length', 'positive_ratio']]
y = df['is_active']

# Преобразование категориальных признаков и нормализация
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'messages_per_day', 'avg_message_length', 'positive_ratio']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['gender', 'education_level'])  # Игнорировать неизвестные категории
    ])

# Создаем модель
model = Sequential([
    Dense(10, activation='relu', input_dim=8),  # input_dim соответствует количеству признаков после предобработки
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Создаем Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
pipeline.fit(X_train, y_train)

# Оценка модели
y_pred = pipeline.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Преобразуем вероятности в метки 0 или 1

# Выводим метрики качества
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Функция для ввода данных пользователем и получения предсказания
def predict_user_activity():
    print("Введите данные пользователя для предсказания активности:")
    # Ввод данных пользователя
    age = int(input("Возраст: "))
    gender = input("Пол (male/female): ").lower()
    education_level = input("Уровень образования (high_school/bachelor/master): ").lower()
    messages_per_day = int(input("Сообщений в день: "))
    avg_message_length = int(input("Средняя длина сообщения (в символах): "))
    positive_ratio = float(input("Доля позитивных сообщений (от 0 до 1): "))

    # Создаем DataFrame для новых данных
    user_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'education_level': [education_level],
        'messages_per_day': [messages_per_day],
        'avg_message_length': [avg_message_length],
        'positive_ratio': [positive_ratio]
    })

    # Получаем предсказание активности
    prediction = pipeline.predict(user_data)
    prediction = (prediction > 0.5).astype(int)  # Преобразуем вероятности в метки 0 или 1

    # Выводим результат
    if prediction == 1:
        print("Пользователь активен.")
    else:
        print("Пользователь не активен.")

# Вызов функции для ввода данных пользователем
predict_user_activity()
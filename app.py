import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

model_save_path = 'models/'

card_data = pd.read_csv('card_transdata.csv')
X = card_data.drop('fraud', axis=1)
y = card_data['fraud']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

def load_models():
    model_ml1 = pickle.load(open(model_save_path + 'model_ml1.pkl', 'rb'))
    model_ml4 = pickle.load(open(model_save_path + 'model_ml4.pkl', 'rb'))
    model_ml5 = pickle.load(open(model_save_path + 'model_ml5.pkl', 'rb'))
    model_ml3 = XGBClassifier()
    model_ml3.load_model(model_save_path + 'model_ml3.json')
    model_ml6 = load_model(model_save_path + 'model_ml6.h5')
    model_ml2 = pickle.load(open(model_save_path + 'kmeans_model.pkl', 'rb'))
    return model_ml1, model_ml3, model_ml4, model_ml5, model_ml6, model_ml2

st.markdown("""
<style>
div[class="css-zbg2rx e1fqkh3o1"] {
    background: url("3R7jLnr3_Gc.jpg");
    background-repeat: no-repeat;
    background-size:350%;
}
h1 {
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

# Сайдбар для навигации
#st.sidebar.image("kawasaki.png", width=100)
st.sidebar.title("Меню")
page = st.sidebar.radio(
    "Выберите страницу:",
    ("Информация о разработчике", "Информация о наборе данных", "Визуализации данных", "Предсказание модели ML")
)


# Функции для каждой страницы
def page_developer_info():
    st.title("Информация о разработчике")
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("Контактная информация")
        st.write("ФИО: Круглов Богдан Евгеньевич")
        st.write("Номер учебной группы: ФИТ-222")
    
    with col2:
        st.header("Фотография")
        st.image("CCI08012024.jpg", width=200)
    
    st.header("Тема РГР")
    st.write("Разработка Web-приложения для инференса моделей ML и анализа данных")

def page_dataset_info():
    st.title("Информация о наборе данных")

    st.markdown("""
    ## Наименование столбцов
    1. distance_from_home
    2. distance_from_last_transaction
    3. ratio_to_median_purchase_price
    4. repeat_retailer
    5. used_chip
    6. used_pin_number
    7. online_order
    8. fraud
    ### distance_from_home (dtype: float)
    Данный столбец характеризует расстояние от дома пользователя в момент
    совершения транзакции
    ### distance_from_last_transaction (dtype: float)
    Данный столбец характеризует расстояние, которое преодолел пользователь
    между совершенными транзакциями
    ### ratio_to_median_purchase_price (dtype: float)
    Данный столбец характеризует отношение между совершенной транзакции к
    средней цене закупки (то есть проверяется насколько текущая трансакция
    аномальна для конкретного пользователя)
    ### repeat_retailer (dtype: float)
    Данный столбец характеризует совершались ли транзакции у данного
    продавца пользователем ранее
    ### used_chip (dtype: float)
    Данный столбец характеризует использовался ли чип карты при совершении
    транзакции (то есть была ли совершенна транзакция физическим способом,
    прикладывая карту к терминалу)
    ### used_pin_number (dtype: float)
    Данный столбец характеризует использовался ли ПИН-код при совершении
    транзакции
    ### online_order (dtype: float)
    Данный столбец характеризует была ли транзакция совершенна онлайн

    ### fraud (dtype: float)
    Данный столбец характеризует была ли текущая операция мошеннической
    ## Описание датасета
    Данный датасет используется в задаче классификации. Модели
    классификации необходимо классифицировать была ли транзакция
    мошеннической. В датасете содержится миллион строк. Датасет был загружен
    на kaggle 2 года назад тех пор не обновлялся.
                
    **Особенности предобработки данных:**
    - Удаление лишних столбцов, например, 'index'.
    - Обработка пропущенных значений.
    - Нормализация числовых данных для улучшения производительности моделей.
    - Кодирование категориальных переменных.
    """)

@st.cache_data
def page_data_visualization():
    st.title('Визуализация данных о транзакциях')

    # Визуализация 1: Расстояние от дома vs Расстояние от последней транзакции
    st.subheader('Расстояние от дома vs Расстояние от последней транзакции')
    with st.spinner('Wait for it...'):
        fig, ax = plt.subplots()
        sns.scatterplot(x='distance_from_home', y='distance_from_last_transaction', hue='fraud', data=card_data, ax=ax)
        st.pyplot(fig)

    # Визуализация 3: Повторный ритейлер
    st.subheader('Повторный ритейлер')
    with st.spinner('Wait for it...'):
        fig, ax = plt.subplots()
        sns.countplot(x='repeat_retailer', hue='fraud', data=card_data, ax=ax)
        st.pyplot(fig)
        
    # Визуализация 4: Использование чипа
    st.subheader('Использование чипа')
    with st.spinner('Wait for it...'):
        fig, ax = plt.subplots()
        sns.countplot(x='used_chip', hue='fraud', data=card_data, ax=ax)
        st.pyplot(fig)
    
    # Визуализация 5: Boxplot
    st.subheader('Boxplot для некатегориальных признаков')
    with st.spinner('Wait for it...'):
        fig, ax = plt.subplots()
        sns.boxplot(data=card_data, y="distance_from_last_transaction")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.boxplot(data=card_data, y="distance_from_home")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.boxplot(data=card_data, y="ratio_to_median_purchase_price")
        st.pyplot(fig)
    

# Функция для загрузки моделей

def load_models():
    model_ml1 = pickle.load(open(model_save_path + 'model_ml1.pkl', 'rb'))
    model_ml4 = pickle.load(open(model_save_path + 'model_ml4.pkl', 'rb'))
    model_ml5 = pickle.load(open(model_save_path + 'model_ml5.pkl', 'rb'))
    model_ml3 = XGBClassifier()
    model_ml3.load_model(model_save_path + 'model_ml3.json')
    model_ml6 = load_model(model_save_path + 'model_ml6.h5')
    model_ml2 = pickle.load(open(model_save_path + 'kmeans_model.pkl', 'rb'))
    return model_ml1, model_ml3, model_ml4, model_ml5, model_ml6, model_ml2

def page_ml_prediction():
    st.title("Предсказания моделей машинного обучения")

    # Виджет для загрузки файла
    uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")

    # Интерактивный ввод данных, если файл не загружен
    if uploaded_file is None:
        st.subheader("Введите данные для предсказания:")
        airlines_data = pd.read_csv('card_transdata.csv')

        # Интерактивные поля для ввода данных
        input_data = {}
        all_columns = airlines_data.columns.tolist()
        feature_names = all_columns
        feature_names.remove('fraud')
        input_data[feature_names[0]] = st.slider(feature_names[0], 0, 10000, 1337, 1)
        input_data[feature_names[1]] = st.slider(feature_names[1], 0, 10000, 1337, 1)
        input_data[feature_names[2]] = st.slider(feature_names[2], 0, 250, 40, 1)
        input_data[feature_names[3]] = st.selectbox(feature_names[3], (0, 1))
        input_data[feature_names[4]] = st.selectbox(feature_names[4], (0, 1))
        input_data[feature_names[5]] = st.selectbox(feature_names[5], (0, 1))
        input_data[feature_names[6]] = st.selectbox(feature_names[6], (0, 1))

        # Загрузка моделей
        model_ml1, model_ml3, model_ml4, model_ml5, model_ml6, model_ml2 = load_models()
        option = st.selectbox(
            "Список моделей для предикта",
            (type(model_ml1).__name__, type(model_ml3).__name__, type(model_ml4).__name__, type(model_ml5).__name__, type(model_ml6).__name__),
            index=None,
            placeholder="Выберите модель для предикта . . .",
            )

        if st.button('Сделать предсказание'):

            input_df = pd.DataFrame([input_data])
            
            st.write("Входные данные:", input_df)

            # Используем масштабировщик, обученный на обучающих данных
            scaler = StandardScaler().fit(X_train)
            scaled_input = scaler.transform(input_df)

            # Делаем предсказания
            prediction_ml1 = model_ml1.predict(scaled_input)
            prediction_ml3 = model_ml3.predict(scaled_input)
            prediction_ml4 = model_ml4.predict(scaled_input)
            prediction_ml5 = model_ml5.predict(scaled_input)
            prediction_ml6 = (model_ml6.predict(scaled_input) > 0.5).astype(int)

            inputer(option, prediction_ml1, prediction_ml3, prediction_ml4, prediction_ml5, prediction_ml6)
    else:
        try:
            model_ml2 = pickle.load(open(model_save_path + 'kmeans_model.pkl', 'rb'))
            model_ml1 = pickle.load(open(model_save_path + 'model_ml1.pkl', 'rb'))
            model_ml4 = pickle.load(open(model_save_path + 'model_ml4.pkl', 'rb'))
            model_ml5 = pickle.load(open(model_save_path + 'model_ml5.pkl', 'rb'))
            model_ml3 = XGBClassifier()
            model_ml3.load_model(model_save_path + 'model_ml3.json')
            model_ml6 = load_model(model_save_path + 'model_ml6.h5')

            # Сделать предсказания на тестовых данных
            cluster_labels = model_ml2.predict(X_test)
            predictions_ml1 = model_ml1.predict(X_test)
            predictions_ml4 = model_ml4.predict(X_test)
            predictions_ml5 = model_ml5.predict(X_test)
            predictions_ml3 = model_ml3.predict(X_test)
            predictions_ml6 = model_ml6.predict(X_test).round() # Округление для нейронной сети

            # Оценить результаты
            rand_score_ml2 = rand_score(y_test, cluster_labels)
            accuracy_ml1 = accuracy_score(y_test, predictions_ml1)
            accuracy_ml4 = accuracy_score(y_test, predictions_ml4)
            accuracy_ml5 = accuracy_score(y_test, predictions_ml5)
            accuracy_ml3 = accuracy_score(y_test, predictions_ml3)
            accuracy_ml6 = accuracy_score(y_test, predictions_ml6)

            st.success(f"rand_score KMeans: {rand_score_ml2}")
            st.success(f"Точность LogisticRegression: {accuracy_ml1}")
            st.success(f"Точность XGBClassifier: {accuracy_ml4}")
            st.success(f"Точность BaggingClassifier: {accuracy_ml5}")
            st.success(f"Точность StackingClassifier: {accuracy_ml3}")
            st.success(f"Точность нейронной сети Tensorflow: {accuracy_ml6}")
        except Exception as e:
            st.error(f"Произошла ошибка при чтении файла: {e}")

def inputer(input_mess, prediction_ml1, prediction_ml3, prediction_ml4, prediction_ml5, prediction_ml6):
    if input_mess == "LogisticRegression":
        st.success(f"Результат предсказания LogisticRegression: {prediction_ml1[0]}")
    elif input_mess == "XGBClassifier":
        st.success(f"Результат предсказания XGBClassifier: {prediction_ml3[0]}")
    elif input_mess == "BaggingClassifier":
        st.success(f"Результат предсказания BaggingClassifier: {prediction_ml4[0]}")
    elif input_mess == "StackingClassifier":
        st.success(f"Результат предсказания StackingClassifier: {prediction_ml5[0]}")
    elif input_mess == "Sequential":
        st.success(f"Результат предсказания Tensorflow: {prediction_ml6[0]}")

if page == "Информация о разработчике":
    page_developer_info()
elif page == "Информация о наборе данных":
    page_dataset_info()
elif page == "Визуализации данных":
    page_data_visualization()
elif page == "Предсказание модели ML":
    page_ml_prediction()

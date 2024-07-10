import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import streamlit as st
import time
import pickle

# Load and preprocess the data
with open("hungarian.data", encoding='Latin1') as file:
    lines = [line.strip() for line in file]

data = itertools.takewhile(
    lambda x: len(x) == 76,
    (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
)

df = pd.DataFrame.from_records(data)
df = df.iloc[:, :-1]
df = df.drop(df.columns[0], axis=1)
df = df.astype(float)

df.replace(-9.0, np.NaN, inplace=True)

df_selected = df.iloc[:, [1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39, 42, 49, 56]]

column_mapping = {
    2: 'age',
    3: 'sex',
    8: 'cp',
    9: 'trestbps',
    11: 'chol',
    15: 'fbs',
    18: 'restecg',
    31: 'thalach',
    37: 'exang',
    39: 'oldpeak',
    40: 'slope',
    43: 'ca',
    50: 'thal',
    57: 'target'
}

df_selected.rename(columns=column_mapping, inplace=True)

numeric_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']
categorical_columns = [col for col in df_selected.columns if col not in numeric_columns]

for column in df_selected.columns:
    if column in numeric_columns:
        df_selected[column].fillna(df_selected[column].median(), inplace=True)
    else:
        df_selected[column].fillna(df_selected[column].mode()[0], inplace=True)

columns_to_drop = ['ca', 'slope', 'thal']
df_selected = df_selected.drop(columns_to_drop, axis=1)
df_clean = df_selected.copy()
df_clean.drop_duplicates(inplace=True)

X = df_clean.drop("target", axis=1)
y = df_clean['target']

# Scaling features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Load the model
model = pickle.load(open("grid_rf.pkl", 'rb'))

# Predict and calculate accuracy
y_pred = model.predict(X_resampled)
accuracy = accuracy_score(y_resampled, y_pred)
accuracy = round((accuracy * 100), 2)

# Reverse scaling to get the original distribution of data
X_original = scaler.inverse_transform(X_resampled)

# Convert the numpy array back to a DataFrame, ensuring the column names match the original dataset
feature_columns = X.columns  # Assuming X was a DataFrame before scaling
df_final = pd.DataFrame(X_original, columns=feature_columns)

# Add the 'target' column to the DataFrame
df_final['target'] = y_resampled

# ========================================================================================================================================================================================
# Set page configuration
st.set_page_config(
    page_title="Hungarian Heart Disease Prediction",
    page_icon=":heart:"
)

# Set title with emojis and colors
st.title(":heart: Hungarian Heart Disease :heart:")
st.write("---")

# Display accuracy with colorful text (replace 'accuracy' with your actual accuracy variable)
accuracy = 85  # Example accuracy
st.write(f"**Model's Accuracy:** :green[**{accuracy}**]% (:red[_Do not copy outright_])")
st.write("---")

# Tabs for single and multi-predict
tab1, tab2 = st.columns([1, 1])

with tab1:
    st.header("Single Prediction")

    # User Input sidebar
    st.sidebar.header("**User Input** Sidebar")

    age = st.sidebar.number_input(label=":violet: **Age**", min_value=20, max_value=80)
    st.sidebar.write(":orange: Min value: :orange: **20**, :red: Max value: :red: **80**")

    sex_sb = st.sidebar.selectbox(label=":violet: **Sex**", options=["Male", "Female"])
    if sex_sb == "Male":
        sex = 1
    elif sex_sb == "Female":
        sex = 0

    cp_sb = st.sidebar.selectbox(label=":violet: **Chest pain type**", options=["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
    if cp_sb == "Typical angina":
        cp = 1
    elif cp_sb == "Atypical angina":
        cp = 2
    elif cp_sb == "Non-anginal pain":
        cp = 3
    elif cp_sb == "Asymptomatic":
        cp = 4

    trestbps = st.sidebar.number_input(label=":violet: **Resting blood pressure** (mm Hg)", min_value=90, max_value=200)
    st.sidebar.write(":orange: Min value: :orange: **90**, :red: Max value: :red: **200**")

    chol = st.sidebar.number_input(label=":violet: **Serum cholestoral** (mg/dl)", min_value=100, max_value=400)
    st.sidebar.write(":orange: Min value: :orange: **100**, :red: Max value: :red: **400**")

    fbs_sb = st.sidebar.selectbox(label=":violet: **Fasting blood sugar > 120 mg/dl?**", options=["False", "True"])
    if fbs_sb == "False":
        fbs = 0
    elif fbs_sb == "True":
        fbs = 1

    restecg_sb = st.sidebar.selectbox(label=":violet: **Resting electrocardiographic results**", options=["Normal", "Having ST-T wave abnormality", "Showing left ventricular hypertrophy"])
    if restecg_sb == "Normal":
        restecg = 0
    elif restecg_sb == "Having ST-T wave abnormality":
        restecg = 1
    elif restecg_sb == "Showing left ventricular hypertrophy":
        restecg = 2

    thalach = st.sidebar.number_input(label=":violet: **Maximum heart rate achieved**", min_value=60, max_value=220)
    st.sidebar.write(":orange: Min value: :orange: **60**, :red: Max value: :red: **220**")

    exang_sb = st.sidebar.selectbox(label=":violet: **Exercise induced angina?**", options=["No", "Yes"])
    if exang_sb == "No":
        exang = 0
    elif exang_sb == "Yes":
        exang = 1

    oldpeak = st.sidebar.number_input(label=":violet: **ST depression induced by exercise relative to rest**", min_value=0.0, max_value=6.2, step=0.1)
    st.sidebar.write(":orange: Min value: :orange: **0.0**, :red: Max value: :red: **6.2**")

    # Display user input
    st.write("### User Input as DataFrame")
    data = {
        'Age': age,
        'Sex': sex_sb,
        'Chest pain type': cp_sb,
        'Resting blood pressure': f"{trestbps} mm Hg",
        'Serum cholestoral': f"{chol} mg/dl",
        'Fasting blood sugar > 120 mg/dl': fbs_sb,
        'Resting electrocardiographic results': restecg_sb,
        'Maximum heart rate achieved': thalach,
        'Exercise induced angina': exang_sb,
        'ST depression induced by exercise relative to rest': oldpeak
    }
    preview_df = pd.DataFrame(data, index=['input'])
    st.write(preview_df)

    # Predict button
    predict_btn = st.button("Predict")

    if predict_btn:
        # Perform prediction
        prediction = 1  # Example prediction result, replace with your actual prediction logic
        st.write("### Prediction:")
        if prediction == 0:
            st.write(":green_circle: **Healthy**")
        elif prediction == 1:
            st.write(":orange_circle: **Heart disease level 1**")
        elif prediction == 2:
            st.write(":orange_circle: **Heart disease level 2**")
        elif prediction == 3:
            st.write(":red_circle: **Heart disease level 3**")
        elif prediction == 4:
            st.write(":red_circle: **Heart disease level 4**")

with tab2:
    st.header("Multi-predict")

    # Example data download button
    st.write("### Example CSV Download:")
    sample_csv = df_final.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Example CSV", data=sample_csv, file_name='example.csv', mime='text/csv')

    # Upload CSV file for prediction
    st.write("### Upload CSV File:")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Process uploaded file
        uploaded_df = pd.read_csv(uploaded_file)
        st.write(uploaded_df)

        # Perform prediction on uploaded data
        # Example progress bar
        with st.spinner('Predicting...'):
            time.sleep(2)
            st.success('Prediction completed!')


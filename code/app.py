import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd  # if you need 


# Load patient outcome data
def load_patient_data():
    # Updated URL to a reliable source
    data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    data = pd.read_csv(url, names=column_names)
    return data



# Define the function to train the model
def train_outcome_model(data):
    # Assume 'target' is the column to predict; replace with the actual target column name
    X = data.drop('column', axis=1)  # Replace 'target' with the correct label column name
    y = data['column']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Logistic Regression model
    model = LogisticRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy


# Call the function and display the accuracy
model, accuracy = train_outcome_model(data)
st.write(f"Patient Outcome Model Accuracy: {accuracy * 100:.2f}%")


# App user interface
st.title("Healthcare Predictive Analytics App")
st.write("Predictive model for patient outcomes based on health metrics.")

# Load data and train model
data = load_patient_data()
model, accuracy = train_outcome_model(data)

st.write(f"Patient Outcome Model Accuracy: {accuracy * 100:.2f}%")

# Interactive prediction form
st.sidebar.header("Enter Patient Details")
def get_user_input():
    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.number_input("Glucose", 0, 200, 120)
    blood_pressure = st.sidebar.number_input("Blood Pressure", 0, 122, 70)
    skin_thickness = st.sidebar.number_input("Skin Thickness", 0, 99, 20)
    insulin = st.sidebar.number_input("Insulin", 0, 846, 79)
    bmi = st.sidebar.number_input("BMI", 0.0, 67.1, 32.0)
    diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.42, 0.5)
    age = st.sidebar.number_input("Age", 0, 120, 33)

    user_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }

    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()

# Display user input
st.subheader("Patient Data")
st.write(user_input)

# Generate prediction
prediction = model.predict(user_input) > 0.5
st.subheader("Prediction")
st.write("The model predicts that the patient has diabetes." if prediction else "The model predicts that the patient does not have diabetes.")


st.write("CREATED by SULEIMAN ADAM")

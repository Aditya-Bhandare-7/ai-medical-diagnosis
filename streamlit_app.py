import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fpdf import FPDF

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Medical Diagnosis Assistant",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 AI Medical Diagnosis Assistant")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["🩺 Diagnosis", "ℹ About Project"])

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("symbipredict_2022.csv")

X = data.drop("prognosis", axis=1)
y = data["prognosis"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(n_estimators=200)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
st.sidebar.success(f"Model Accuracy: {accuracy*100:.2f}%")

# -----------------------------
# Sidebar Patient Profile
# -----------------------------
st.sidebar.header("👤 Patient Profile")

name = st.sidebar.text_input("Patient Name")
age = st.sidebar.slider("Age", 1, 100, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
weight = st.sidebar.number_input("Weight (kg)", 20, 200, 60)

existing_conditions = st.sidebar.multiselect(
    "Existing Medical Conditions",
    ["Diabetes", "Hypertension", "Asthma", "Heart Disease", "None"]
)

# -----------------------------
# Disease Info Function
# -----------------------------
def get_disease_info(disease):

    database = {
        "Fungal infection": {
            "description": "A fungal infection affecting skin, hair or nails.",
            "precautions": [
                "Keep skin dry",
                "Maintain hygiene",
                "Avoid sharing clothes",
                "Use antifungal medication"
            ],
            "doctor": "Dermatologist"
        },

        "Allergy": {
            "description": "An immune response to allergens such as dust, pollen or food.",
            "precautions": [
                "Avoid allergens",
                "Keep environment clean",
                "Use antihistamines",
                "Wear mask in dusty areas"
            ],
            "doctor": "Allergist"
        },

        "Diabetes": {
            "description": "A chronic condition affecting blood sugar regulation.",
            "precautions": [
                "Monitor blood sugar",
                "Exercise regularly",
                "Eat healthy food",
                "Avoid excess sugar"
            ],
            "doctor": "Endocrinologist"
        }
    }

    if disease in database:
        return database[disease]

    return {
        "description": f"{disease} is a medical condition that should be evaluated by a healthcare professional.",
        "precautions": [
            "Consult a doctor",
            "Get adequate rest",
            "Maintain hygiene",
            "Eat balanced diet"
        ],
        "doctor": "General Physician"
    }

# -----------------------------
# PDF Generator
# -----------------------------
def generate_pdf(name, age, gender, weight, disease, confidence, doctor):

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=16)
    pdf.cell(200,10,"AI Medical Diagnosis Report",ln=True,align="C")

    pdf.set_font("Arial", size=12)

    pdf.cell(200,10,f"Patient Name: {name}",ln=True)
    pdf.cell(200,10,f"Age: {age}",ln=True)
    pdf.cell(200,10,f"Gender: {gender}",ln=True)
    pdf.cell(200,10,f"Weight: {weight} kg",ln=True)

    pdf.cell(200,10,"",ln=True)

    pdf.cell(200,10,f"Predicted Disease: {disease}",ln=True)
    pdf.cell(200,10,f"Confidence: {confidence:.2f}%",ln=True)
    pdf.cell(200,10,f"Recommended Doctor: {doctor}",ln=True)

    pdf.cell(200,10,"",ln=True)
    pdf.cell(200,10,"This report is AI generated for educational purposes.",ln=True)

    pdf.output("report.pdf")

# -----------------------------
# Diagnosis Tab
# -----------------------------
with tab1:

    st.subheader("🩺 Select Symptoms")

    symptom_list = X.columns.tolist()

    selected_symptoms = st.multiselect(
        "Choose your symptoms",
        symptom_list
    )

    input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]

    if st.button("🔍 Predict Disease"):

        if len(selected_symptoms) == 0:
            st.warning("Please select at least one symptom.")

        else:

            prediction = model.predict([input_data])
            probability = model.predict_proba([input_data])

            disease = encoder.inverse_transform(prediction)[0]
            confidence = max(probability[0]) * 100

            info = get_disease_info(disease)

            st.divider()

            st.markdown("## 🧬 Diagnosis Result")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Name:**", name)
                st.write("**Age:**", age)

            with col2:
                st.write("**Gender:**", gender)
                st.write("**Weight:**", weight)

            st.success(f"Predicted Disease: {disease}")
            st.info(f"Confidence: {confidence:.2f}%")

            with st.expander("🩺 Disease Description"):
                st.write(info["description"])

            with st.expander("🛡 Precautions"):
                for p in info["precautions"]:
                    st.write("•", p)

            st.success(f"👨‍⚕ Recommended Specialist: {info['doctor']}")

            st.divider()

            prob_df = pd.DataFrame({
                "Disease": encoder.classes_,
                "Probability": probability[0]
            }).sort_values(by="Probability", ascending=False).head(5)

            st.subheader("📊 Top 5 Possible Diseases")

            st.bar_chart(prob_df.set_index("Disease"))

            # Generate PDF
            generate_pdf(name, age, gender, weight, disease, confidence, info["doctor"])

            with open("report.pdf", "rb") as f:
                st.download_button(
                    "📄 Download Medical Report",
                    f,
                    file_name="medical_report.pdf"
                )

            st.link_button(
                "🏥 Find Nearby Hospitals",
                "https://www.google.com/maps/search/hospitals+near+me"
            )

# -----------------------------
# About Project Tab
# -----------------------------
with tab2:

    st.subheader("About This Project")

    st.write("""
This **AI Medical Diagnosis Assistant** predicts possible diseases based on symptoms using Machine Learning.

### Technologies Used
• Python  
• Streamlit  
• Scikit-Learn  
• Random Forest Algorithm  

### Features
• Symptom-based disease prediction  
• Patient profile input  
• Probability visualization  
• AI generated medical report (PDF)  

This project was developed as an **engineering AI mini project**.
""")

# -----------------------------
# Footer
# -----------------------------
st.warning(
"⚠ This system is for educational purposes only and not a substitute for professional medical advice."
)

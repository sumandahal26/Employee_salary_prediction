import streamlit as st
import pandas as pd 
import joblib 

model = joblib.load("placement_gbm_model.pkl")


scaler = joblib.load("scaler.pkl")
branch_encoder = joblib.load("branch_encoder.pkl")


branch_opts = branch_encoder.classes_

st.set_page_config(page_title="Fresher Salary Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Employee( Fresher) Placement Salary Predictor")
st.markdown("Hello 👋!!! Welcome 🙏 .This app helps to predict the estimated placement package for college freshers.")
st.markdown("Just enter some details, and our model will try to predict the salary as your qualities !")

st.sidebar.header("Input Student Details")


age = st.sidebar.slider("Age of Candidate", 18, 30, 21) 
dsa_q = st.sidebar.slider("No. of DSA questions", 0, 500, 100)
cgpa = st.sidebar.slider("CGPA (out of 10)", 0.0, 10.0, 7.5, 0.1)
backlogs = st.sidebar.slider("No. of backlogs", 0, 10, 0)

branch = st.sidebar.selectbox("Branch of Engineering", branch_opts)


knows_ml = st.sidebar.checkbox("Knows ML?")
knows_dsa = st.sidebar.checkbox("Knows DSA?")
knows_python = st.sidebar.checkbox("Knows Python?")
knows_js = st.sidebar.checkbox("Knows JavaScript?")


expected_feat_order = [
    'No. of DSA questions', 'CGPA', 'Knows ML', 'Knows DSA', 'Knows Python',
    'Knows JavaScript', 'No. of backlogs', 'Age of Candidate', 'Branch of Engineering'
]

input_data_dict = {
    'No. of DSA questions': dsa_q,
    'CGPA': cgpa,
    'Knows ML': 1 if knows_ml else 0,
    'Knows DSA': 1 if knows_dsa else 0,
    'Knows Python': 1 if knows_python else 0,
    'Knows JavaScript': 1 if knows_js else 0,
    'No. of backlogs': backlogs,
    'Age of Candidate': age,
    'Branch of Engineering': branch
}


input_df = pd.DataFrame([input_data_dict], columns=expected_feat_order)


input_df['Branch of Engineering'] = branch_encoder.transform(input_df['Branch of Engineering'])


num_cols_for_scaling = [
    'No. of DSA questions', 'CGPA', 'No. of backlogs', 'Age of Candidate'
]
input_df[num_cols_for_scaling] = scaler.transform(input_df[num_cols_for_scaling])


st.write("### 🔎 Input Data (Processed for Model)")
st.write(input_df)


if st.button("Predict Placement Package"):
    prediction = model.predict(input_df)
    st.success(f"🎉 Congratulations Your Estimated Placement Package: ₹{prediction[0]:.2f} LPA")
    st.balloons() 

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    batch_data_for_model = batch_data.copy()
    
   
    cols_to_drop_from_batch_for_model = [
        'Name of Student', 'Roll No.',
        'Interview Room Temperature', 'Knows Cricket', 'Knows Dance',
        'Participated in College Fest', 'Was in Coding Club', 'Knows HTML', 'Knows CSS',
        'Placement Package' 
    ]
    
   
    for col in cols_to_drop_from_batch_for_model:
        if col in batch_data_for_model.columns:
            batch_data_for_model = batch_data_for_model.drop(columns=[col])

    
    yn_cols_batch = ['Knows ML', 'Knows DSA', 'Knows Python', 'Knows JavaScript']
    for col in yn_cols_batch:
        
        batch_data_for_model[col] = batch_data_for_model[col].map({'Yes': 1, 'No': 0})

  
    batch_data_for_model = batch_data_for_model[(batch_data_for_model['Age of Candidate'] >= 18) & (batch_data_for_model['Age of Candidate'] <= 30)]


    batch_data_for_model.drop_duplicates(inplace=True)

    
    batch_data_for_model['Branch of Engineering'] = branch_encoder.transform(batch_data_for_model['Branch of Engineering'])

    
    num_cols_for_scaling_batch = [
        'No. of DSA questions', 'CGPA', 'No. of backlogs', 'Age of Candidate'
    ]
    batch_data_for_model[num_cols_for_scaling_batch] = scaler.transform(batch_data_for_model[num_cols_for_scaling_batch])

    
    for col in expected_feat_order:
        if col not in batch_data_for_model.columns:
            batch_data_for_model[col] = 0 
            
    batch_data_final_for_model = batch_data_for_model[expected_feat_order]

    st.write("Processed data for batch prediction (first 5 rows):")
    st.write(batch_data_final_for_model.head())

    batch_preds = model.predict(batch_data_final_for_model)
    batch_data['Predicted Placement Package (LPA)'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_packages.csv', mime='text/csv')

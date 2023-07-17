import pickle
import streamlit as st

model = pickle.load(open('home_loan_approval_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def preprocess_data(data):
    gender_encoded = 1 if data['Gender'] == "Male" else 0
    marital_status_encoded = 1 if data['Marital Status'] == "Married" else 0
    education_encoded = 1 if data['Education'] == "Graduate" else 0
    self_employed_encoded = 1 if data['Self Employed'] == "Yes" else 0
    property_area_encoded = {"Semi-Urban": 1, "Urban": 0, "Rural": 2}[data['Property Area']]

    processed_data = {
        'Gender': gender_encoded,
        'Marital Status': marital_status_encoded,
        'Dependents': data['Dependents'],
        'Education': education_encoded,
        'Self Employed': self_employed_encoded,
        'Loan Amount': data['Loan Amount'],
        'Loan Amount Term': data['Loan Amount Term'],
        'Credit History': data['Credit History'],
        'Property Area': property_area_encoded,
        'Income': data['Income']
    }

    return processed_data

def main():
    st.title("Home Loan Approval Prediction")
    image = 'home_loan_approval_image.jpg'  # Path to your image file
    st.image(image, caption='Home Loan Approval', use_column_width=True)
    st.subheader("Get Your Dream Home Approved")
    st.write("Our advanced model can predict home loan approvals accurately. Enter the required details below and let us help you in achieving your dream of owning a home.")


    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    dependents = int(st.number_input("Dependents", value=0))
    income = st.number_input("Income")
    loan_amount = st.number_input("Loan Amount")
    loan_amount_term = st.number_input("Loan Amount Term", min_value=12, help="Enter the loan term in months")
    credit_history = st.selectbox("Credit History", [0, 1])
    property_area = st.selectbox("Property Area", ["Semi-Urban", "Urban", "Rural"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

    if st.button("Predict"):
        input_data = {
            'Gender': gender,
            'Marital Status': marital_status,
            'Dependents': dependents,
            'Education': education,
            'Self Employed': self_employed,
            'Loan Amount': loan_amount,
            'Loan Amount Term': loan_amount_term,
            'Credit History': credit_history,
            'Property Area': property_area,
            'Income': income
        }

        processed_data = preprocess_data(input_data)

        # Scale specific columns
        columns_to_scale = ['Income', 'Loan Amount', 'Loan Amount Term']
        scaled_data = scaler.transform([[processed_data[col] for col in columns_to_scale]])

        # Update the processed data with scaled values
        for i, col in enumerate(columns_to_scale):
            processed_data[col] = scaled_data[0][i]

        prediction = model.predict([[processed_data[col] for col in processed_data]])

        if prediction[0] == 1:
            st.success("Congratulations! Your loan has been approved. You're one step closer to realizing your dreams. Enjoy your new home!")
        else:
            st.error("We regret to inform you that your loan application has been denied. We understand that this may be disappointing, but we encourage you to explore alternative options and keep pursuing your goals. Best of luck in your future endeavors.")

if __name__ == '__main__':
    main()
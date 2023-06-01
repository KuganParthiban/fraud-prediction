import numpy as np
import pickle
import streamlit as st

# Loading the model
loaded_model = pickle.load(open('fraud_predict.pkl', 'rb'))

# Loading the label encoder mappings
label_encoder_mappings = pickle.load(open('label_encoder_mappings2.pkl', 'rb'))

# Creating a function for prediction
def churn_prediction(input_data):
    # Convert categorical features using Label Encoding
    for feature in label_encoder_mappings:
        label_encoder = label_encoder_mappings[feature]
        input_data[feature] = label_encoder.fit_transform([input_data[feature]])

    # Convert input data to numpy array
    input_data_as_numpy_array = np.array(list(input_data.values()), dtype=object)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'Not Fraud'
    else:
        return 'Fraud Transaction'


def main():
    # giving a title
    st.title('Bank Fraud Detection Web App')

    # getting the input data from the user
    Transaction_Type = st.sidebar.selectbox('How did you transfer?', ('CASH_OUT', 'CASH_IN', 'TRANSFER', 'DEBIT'), key="transaction_type_key")
    Amount = st.sidebar.slider('Amout Transacted', 0, 10000000, 233645)
    initial_balance = st.sidebar.slider('Sender initial balance', 0, 38939424, 1397)
    new_balance = st.sidebar.slider('Sender new balance', 0, 38946233, 1275517)
    recipient_initial_balance = st.sidebar.slider('Recipient initial balance', 0, 42283775, 1496481)
    recipient_new_balance = st.sidebar.slider('Recipient new balance', 0, 42655769, 1666233)

    # code for prediction
    diagnosis = ''

    # creating a button for prediction
    if st.button('Churn Result'):
        input_data = {
            'Transaction_Type': Transaction_Type,
            'Amount': Amount,
            'initial_balance': initial_balance,
            'new_balance': new_balance,
            'recipient_initial_balance': recipient_initial_balance,
            'recipient_new_balance': recipient_new_balance
           
        }
        diagnosis = churn_prediction(input_data)

    st.success(diagnosis)

if __name__ == '__main__':
    main()
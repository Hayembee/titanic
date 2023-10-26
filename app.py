import streamlit as st
import pandas as pd
import pickle

# Load the trained RandomForestClassifier
with open('RandomForestClassifier.pkl', 'rb') as pickle_in:
    rfc = pickle.load(pickle_in)

# Streamlit App
st.title('Titanic Survival Prediction')

# Sidebar for user input
selected_Pclass = st.sidebar.selectbox('Select Ticket Class:', ['Upper class', 'Middle class', 'Lower class'])
selected_sex = st.sidebar.selectbox('Select Gender:', ['Male', 'Female'])
selected_SibSp = st.sidebar.number_input('Number of Siblings/Spouses Aboard:', value=0, step=1)
selected_Parch = st.sidebar.number_input('Number of Parents/Children Aboard:', value=0, step=1)
selected_Embarked = st.sidebar.selectbox('Select Port of Embarkation where C is Cherbough, Q is Queenstown, S is Southampton:', ['C', 'Q', 'S'])

# Map categorical inputs to numerical values
sex_mapping = {'Male': 0, 'Female': 1}
embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
Pclass_mapping = {'Upper class': 1, 'Middle class': 2, 'Lower class': 3}

selected_sex_encoded = sex_mapping[selected_sex]
selected_Embarked_encoded = embarked_mapping[selected_Embarked]
selected_Pclass_encoded = Pclass_mapping[selected_Pclass]

# Prepare input for prediction
input_data = pd.DataFrame({
    'Pclass': [selected_Pclass_encoded],
    'Sex': [selected_sex_encoded],
    'SibSp': [selected_SibSp],
    'Parch': [selected_Parch],
    'Embarked': [selected_Embarked_encoded]
})

# Predict function
def predict_survival(input_data):
    prediction = rfc.predict(input_data)[0]
    return prediction

# Display prediction button
if st.button('Predict'):
    prediction_result = predict_survival(input_data)

    # Display prediction
    if prediction_result == 0:
        st.write('If you go on the titanic ship, you will not survive on the Titanic.')
    else:
        st.write('If you go on the titanic ship, you will survive on the Titanic.')

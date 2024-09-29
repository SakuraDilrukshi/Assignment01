import streamlit as st
import pandas as pd 
import plotly.express as px 
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import pickle
import streamlit as st
import time

app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction']) #two pages
if app_mode=='Home':    
    
    st.title("Welcome to the Diabetes Predictor.")

    #image_path = "D:\\Documents\\MachineLearning02\\App01\\img.jpeg"
    #st.image(image_path)
    st.write("This interactive application is designed to help you explore and understand the dataset from the Nation Institute of")
    st.write("Below diagrams visualize the factots of distribution in the dataset.\n\n")
    df= pd.read_csv("diabetes.csv")
    #st.write(df)

    # Bar Plot for Outcome Counts
    st.subheader('Distribution of Diabetes Outcomes')
    fig, ax = plt.subplots()
    sns.countplot(x='Outcome', data=df, ax=ax)
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - 0.3
        # we change the bar width
        patch.set_width(0.3)
        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)
    ax.set_title('Distribution of Diabetes Outcomes')
    ax.set_xlabel('Diabetes Status')
    ax.set_ylabel('Number of Individuals')
    ax.set_xticklabels(['No Diabetes', 'Diabetes'])
    st.pyplot(fig)
    st.write("")
    st.write("")
    st.write("")

    #Age Distribution
    st.subheader('Age Distribution')
    fig, ax = plt.subplots()
    sns.histplot(df['Age'], kde= True, bins=20, ax=ax)
    ax.set_title('Age Ditribution of Individuals')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    st.write("")
    st.write("")
    st.write("")

    #BMI Distribution
    st.subheader('BMI Distribution')
    fig, ax = plt.subplots()
    sns.histplot(df['BMI'], kde= True, bins=20, ax=ax)
    ax.set_title('BMI Ditribution of Individuals')
    ax.set_xlabel('BMI')
    ax.set_ylabel('Frequency')
    ax.set_xticklabels(['No Diabetes', 'Diabetes'])
    st.pyplot(fig)
    st.write("")
    st.write("")
    st.write("")

    #Glucose Distribution
    st.subheader('Glucose Distribution')
    fig, ax = plt.subplots()
    sns.histplot(df['Glucose'], kde= True, bins=20, ax=ax)
    ax.set_title('Glucose Ditribution of Individuals')
    ax.set_xlabel('Glucose Level')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    st.write("")
    st.write("")
    st.write("")

    #Insulin Distribution
    st.subheader('Insulin Distribution')
    fig, ax = plt.subplots()
    sns.histplot(df['Insulin'], kde= True, bins=20, ax=ax)
    ax.set_title('Insulin Ditribution of Individuals')
    ax.set_xlabel('Insulin Level')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    st.write("")
    st.write("")
    st.write("")

    #Correlation Heatmap
    st.subheader('Correlation Heatmap')
    fig, ax = plt.subplots(figsize=(10,8))
    corr = df.corr()
    sns.heatmap(corr, annot=True,  cmap ='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap of Features')
    st.pyplot(fig)
    st.write("")
    st.write("")
    st.write("")

    #Box Plot for multiple features
    st.subheader('Box Plot of Features')
    fig, ax = plt.subplots(figsize=(12,8))
    sns.boxplot(data=df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']], ax=ax)
    ax.set_title('Box Plot of Selected Features')
    ax.set_xlabel('Features')
    ax.set_ylabel('Value')
    st.pyplot(fig)
    st.write("")
    st.write("")
    st.write("")

    st.sidebar.success("Select Pages From Above Menue")

elif app_mode == 'Prediction':


    loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))

    def diabetes_prediction(input_data):
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = loaded_model.predict(input_data_reshaped)
        print(prediction)
        if (prediction[0] == 0):
            return 'The person is not diabetic'
        else:
            return 'The person is diabetic'

    def main():
        st.title('Diabetes Prediction Web App')

        pregnancies = st.text_input('Number of Pregnancies')
        glucose = st.text_input('Glucose Level')
        bp = st.text_input('Blood Pressure Value')
        skin_thickness = st.text_input('Skin Thickness Value')
        insulin = st.text_input('Insulin Level')
        bmi = st.text_input('BMI Value')
        dpf = st.text_input('Diabetes Pedigree Function Value')
        age = st.text_input('Age of the Person')

        diagnosis = ''
        
        if st.button('Diabetes Test Result'):
            with st.spinner('Please wait...'):
                time.sleep(2)
                diagnosis = diabetes_prediction([pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age])
            
        st.success(diagnosis)

    if __name__ == '__main__':
        main()






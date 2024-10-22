import pandas as pd
import plotly.express as px
import streamlit as st

st.title("Titanic Data Analysis")
st.write("This app analyzes the Titanic dataset and displays various visualizations.")

@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    data = pd.read_csv(url)
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data.drop('Cabin', axis=1, inplace=True)
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = data['FamilySize'].apply(lambda x: 1 if x == 1 else 0)
    return data

data = load_data()

if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(data)

gender_survival = data.groupby('Sex')['Survived'].mean().reset_index()
fig_gender_survival = px.bar(gender_survival, x='Sex', y='Survived', title='Survival Rate by Gender')
st.plotly_chart(fig_gender_survival)

pclass_survival = data.groupby('Pclass')['Survived'].mean().reset_index()
fig_pclass_survival = px.bar(pclass_survival, x='Pclass', y='Survived', title='Survival Rate by Passenger Class')
st.plotly_chart(fig_pclass_survival)

fig_age_distribution = px.histogram(data, x='Age', nbins=50, title='Age Distribution of Passengers')
st.plotly_chart(fig_age_distribution)

fig_age_survival = px.histogram(data, x='Age', color='Survived', nbins=50, title='Age Distribution by Survival Status')
st.plotly_chart(fig_age_survival)

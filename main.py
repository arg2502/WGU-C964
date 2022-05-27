import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split

url = "data/winequality-red.csv"
df = pd.read_csv(url)


def get_data_from_df(dataframe, target_column):
    df_x = dataframe.drop(target_column, axis=1)
    df_y = dataframe[target_column]

    return df_x, df_y


X, y = get_data_from_df(df, "quality")


# TESTING OUT STREAMLIT

header = st.container()
dataset = st.container()
modelTraining = st.container()




with header:
    st.title('Welcome to my awesome data science project')
    st.text("Let's take a look at some wine")

with dataset:
    st.header('Red wine dataset')

    wines = pd.read_csv(url)
    st.write(wines.head())

    st.subheader("Fixed Acidity")
    fixed_acidity_count = pd.DataFrame(wines["fixed acidity"].value_counts().head(15))
    st.bar_chart(fixed_acidity_count)


with modelTraining:
    st.header('Train the model')

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider("What should be the max_depth of the model?", min_value=10, max_value=100, value=20, step=10)
    n_estimators = sel_col.selectbox("How many trees?", options=[100, 200, 300, "No Limit"], index=0)
    input_feature = sel_col.text_input('Which feature should be used as the input?', "citric acid")

    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    # X = wines[[input_feature]]
    # y = wines["quality"]
    np.random.seed(25)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    disp_col.subheader("Accuracy score of the model is:")
    disp_col.write(accuracy_score(y_test, prediction))

    disp_col.subheader("Mean Absolute Error of the model is:")
    disp_col.write(mean_absolute_error(y_test, prediction))

    disp_col.subheader("Mean Squared Error of the model is:")
    disp_col.write(mean_squared_error(y_test, prediction))

    disp_col.subheader("R Squared Score of the model is:")
    disp_col.write(r2_score(y_test, prediction))

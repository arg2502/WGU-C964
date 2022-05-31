import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from imblearn.over_sampling import RandomOverSampler

# Set random seed to get same results every time
np.random.seed(25)

# Load in the raw csv datafile
url = "data/winequality-red.csv"
wines = pd.read_csv(url)

# Separate file into input (X) and output (y)
X = wines.drop("quality", axis=1)
y = wines["quality"]

# Over sample: adds duplicates of the minority qualities for a more balanced (and accurate) prediction
rs = RandomOverSampler()
X, y = rs.fit_resample(X, y)

# Split data into training and testing sets to evaluate model's performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Check the performance of the model, how accurate is it
clf_y_pred = clf.predict(X_test)
clf_acc_score = accuracy_score(y_test, clf_y_pred)
clf_cross_val = np.mean(cross_val_score(clf, X, y, cv=5))

# Streamlit containers
header = st.container()
piegraph = st.container()
histogram = st.container()
scatterplot = st.container()
guesstimator = st.container()

with header:
    st.title("Quality Analysis of Wine")
    st.subheader("A look at the different properties of wine and attempting to predict the quality.")


with piegraph:

    st.title("Number of wines per quality")

    fig, ax = plt.subplots()

    qualities = np.unique(wines["quality"])

    fracs = []
    for q in qualities:
        quantity = len(wines[wines["quality"] == q])
        fracs.append(quantity)
    ax.pie(fracs, labels=qualities, startangle=0, shadow=True, autopct='%1.1f%%')

    st.pyplot(fig)

with histogram:
    st.title("Comparing Qualities")

    fig, axs = plt.subplots(len(X.columns), 2, tight_layout=True, figsize=(10, 30))

    good_wines = wines[wines["quality"] > 5]
    bad_wines = wines[wines["quality"] <= 5]
    num_bins = 30
    axs[0][0].set_title("Good Wines (Quality: 6 and up)")
    axs[0][1].set_title("Poor Wines (Quality: 5 and below)")
    index = 0
    for col in X.columns:
        hist0 = axs[index][0].hist(good_wines[col], bins=num_bins)
        axs[index][0].set_ylabel(col)
        hist1 = axs[index][1].hist(bad_wines[col], bins=num_bins)
        x0_lowerlim, x0_upperlim = axs[index][0].get_xlim()
        x1_lowerlim, x1_upperlim = axs[index][1].get_xlim()

        x_lowerlim = min(x0_lowerlim, x1_lowerlim)
        x_upperlim = min(x0_upperlim, x1_upperlim)

        y_lowerlim = 0
        y_upperlim = max(hist0[0].max(), hist1[0].max())

        axs[index][0].set_xlim([x_lowerlim, x_upperlim])
        axs[index][1].set_xlim([x_lowerlim, x_upperlim])
        axs[index][0].set_ylim([y_lowerlim, y_upperlim])
        axs[index][1].set_ylim([y_lowerlim, y_upperlim])
        index += 1

    st.pyplot(fig)

with scatterplot:
    st.title("Comparing Properties")
    sel_col, display_col = st.columns([1, 3])

    x_axis = sel_col.selectbox("X-Axis", options=X.columns, index=0)
    y_axis = sel_col.selectbox("Y-Axis", options=X.columns, index=0)

    fig, ax = plt.subplots()

    scatter = ax.scatter(x=wines[x_axis],
                         y=wines[y_axis],
                         c=wines["quality"])

    ax.set(xlabel=x_axis, ylabel=y_axis)

    legend = ax.legend(*scatter.legend_elements(),
                       loc="upper right",
                       title="Quality"
                       )
    ax.add_artist(legend)

    display_col.pyplot(fig)

with guesstimator:
    st.title("Predicting the Quality")

    cols = st.columns(3)

    index = 0
    col_maxs = []
    for c in X.columns:
        min_val = wines[c].min()
        max_val = wines[c].max()

        starting_val = min_val + ((max_val - min_val) / 2.0)
        c_max = cols[index].slider(c, min_value=min_val, max_value=max_val, step=0.1, value=float(starting_val))
        col_maxs.append(c_max)
        if index >= 2:
            index = 0
        else:
            index += 1

    fixed_aciditiy, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, \
        total_sulfur_dioxide, density, pH, sulphates, alcohol = col_maxs

    custom_prediction = clf.predict([[fixed_aciditiy, volatile_acidity, citric_acid,
                                      residual_sugar, chlorides, free_sulfur_dioxide,
                                      total_sulfur_dioxide, density, pH,
                                      sulphates, alcohol]])

    st.text(f"Predicted Quality: {custom_prediction}")
    st.text(f"Accuracy Score: {clf_acc_score * 100.0:.2f}% | Cross Val Score: {clf_cross_val * 100.0:.2f}%")

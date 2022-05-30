import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


np.random.seed(25)

url = "data/winequality-red.csv"
wines = pd.read_csv(url)

X = wines.drop("quality", axis=1)
y = wines["quality"]

# Over sample: adds duplicates of the minority qualities for a more balanced (and accurate) prediction
rs = RandomOverSampler()
X, y = rs.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

st_x = StandardScaler()
X_train = st_x.fit_transform(X_train)
X_test = st_x.transform(X_test)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
clf_y_pred = clf.predict(X_test)
clf_acc_score = accuracy_score(y_test, clf_y_pred)
clf_cross_val = np.mean(cross_val_score(clf, X, y, cv=5))

# Streamlit
header = st.container()
piegraph = st.container()
histogram = st.container()
scatterplot = st.container()
guesstimator = st.container()

with header:
    st.title("Hello There")

with piegraph:

    st.title("Number of wines per quality")

    fig, ax = plt.subplots()

    qualities = np.unique(wines["quality"])

    fracs = []
    for q in qualities:
        quantity = len(wines[wines["quality"] == q])
        st.text(f"quantity for {q}: {quantity}")
        fracs.append(quantity)
    st.text(f"fracs: {fracs}")
    ax.pie(fracs, labels=qualities, startangle=0, shadow=True, autopct='%1.1f%%')

    st.pyplot(fig)

with histogram:
    st.title("Comparing Qualities")
    # st.text("Splitting the data into good wines and poor wines and comparing the different properties.")

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
        # st.text(f"{col} 0 max: {hist0[0].max()} | 1 max: {hist1[0].max()}")
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
    # col_maxs = np.empty(len(X.columns))
    col_maxs = []
    for c in X.columns:
        min_val = wines[c].min()
        max_val = wines[c].max()

        starting_val = min_val + ((max_val - min_val) / 2.0)
        # st.text(f"min: {min_val} | max: {max_val} | starting_val: {starting_val}")
        c_max = cols[index].slider(c, min_value=min_val, max_value=max_val, step=0.1, value=float(starting_val))
        col_maxs.append(c_max)
        if index >= 2:
            index = 0
        else:
            index += 1

    # fixed_acidity = cols[0].slider("fixed acidity", min_value=0, max_value=wines["fixed acidity"].max())
    fixed_aciditiy, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol = col_maxs

    custom_prediction = clf.predict([[fixed_aciditiy, volatile_acidity, citric_acid,
                                      residual_sugar, chlorides, free_sulfur_dioxide,
                                      total_sulfur_dioxide, density, pH,
                                      sulphates, alcohol]])

    test_prediction = clf.predict([[8.5,0.28,0.56,1.8,0.092,35.0,103.0,0.9969,3.3,0.75,10.5]])

    st.text(f"Predicted Quality: {custom_prediction}")
    # st.text(f"test prediction: {test_prediction}")

    st.text(f"Accuracy Score: {clf_acc_score} | Cross Val Score: {clf_cross_val}")

    # qualities = np.unique(wines["quality"])
    # st.text(qualities)
    #
    # def get_col_condition(col_nm):
    #     min_val = wines["fixed acidity"].min()
    #     max_val = wines["fixed acidity"].max()
    #     mid_val = min_val + ((max_val - min_val) / 2.0)
    #
    #     fa_toggle = sel_col.select_slider(col_nm, ["Low", "High"])
    #     condition = ""
    #     if fa_toggle == "Low":
    #         condition = wines[col_nm] <= mid_val
    #     else:
    #         condition = wines[col_nm] > mid_val
    #
    #     return condition
    #
    # def filter_wines():
    #     fa_condition = get_col_condition("fixed acidity")
    #     va_condition = get_col_condition("volatile acidity")
    #     ca_condition = get_col_condition("citric acid")
    #     rs_condition = get_col_condition("residual sugar")
    #     chlorides_condition = get_col_condition("chlorides")
    #     fsd_condition = get_col_condition("free sulfur dioxide")
    #     tsd_condition = get_col_condition("total sulfur dioxide")
    #     density_condition = get_col_condition("density")
    #     ph_condition = get_col_condition("pH")
    #     sulphates_condition = get_col_condition("sulphates")
    #     alcohol_condition = get_col_condition("alcohol")
    #
    #     # return np.where(fa_condition &
    #     return wines.where(fa_condition &
    #                        va_condition &
    #                        ca_condition &
    #                        rs_condition &
    #                        chlorides_condition &
    #                        fsd_condition &
    #                        tsd_condition &
    #                        density_condition &
    #                        ph_condition &
    #                        sulphates_condition &
    #                        alcohol_condition).dropna()
    #
    # # filtered_wines = wines[wines["quality"] == 5]
    # # filtered_wines = np.where((wines["quality"] == 5) & (wines["quality"] == 6))
    # filtered_wines = filter_wines()
    # display_col.text(filtered_wines)
    # fracs = np.array([
    #     # filtered_wines[filtered_wines["quality"] == 1].size,
    #     # filtered_wines[filtered_wines["quality"] == 2].size,
    #     filtered_wines[filtered_wines["quality"] == 3].size,
    #     filtered_wines[filtered_wines["quality"] == 4].size,
    #     filtered_wines[filtered_wines["quality"] == 5].size,
    #     filtered_wines[filtered_wines["quality"] == 6].size,
    #     filtered_wines[filtered_wines["quality"] == 7].size,
    #     filtered_wines[filtered_wines["quality"] == 8].size,
    #     # filtered_wines[filtered_wines["quality"] == 9].size,
    #     # filtered_wines[filtered_wines["quality"] == 10].size,
    # ])
    #
    # # explode = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    #
    # ax.pie(fracs, labels=qualities, startangle=90, shadow=True)
    # display_col.pyplot(fig)

    # it's working....but it doesn't really display much
    # try to rethink this
    # maybe it can still be another type of chart?
    # maybe it can display a percentage of the whole?

# # TESTING OUT STREAMLIT
#
# header = st.container()
# dataset = st.container()
# modelTraining = st.container()
#
#
# with header:
#     st.title('Welcome to my awesome data science project')
#     st.text("Let's take a look at some wine")
#
# with dataset:
#     st.header('Red wine dataset')
#
#     wines = pd.read_csv(url)
#     st.write(wines.head())
#
#     st.subheader("Fixed Acidity")
#     fixed_acidity_count = pd.DataFrame(wines["fixed acidity"].value_counts().head(15))
#     st.bar_chart(fixed_acidity_count)
#
#
# with modelTraining:
#     st.header('Train the model')
#
#     sel_col, disp_col = st.columns(2)
#
#     max_depth = sel_col.slider("What should be the max_depth of the model?", min_value=10, max_value=100, value=20, step=10)
#     n_estimators = sel_col.selectbox("How many trees?", options=[100, 200, 300, "No Limit"], index=0)
#     input_feature = sel_col.text_input('Which feature should be used as the input?', "citric acid")
#
#     clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
#     # X = wines[[input_feature]]
#     # y = wines["quality"]
#     np.random.seed(25)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
#     clf.fit(X_train, y_train)
#     prediction = clf.predict(X_test)
#
#     disp_col.subheader("Accuracy score of the model is:")
#     disp_col.write(accuracy_score(y_test, prediction))
#
#     disp_col.subheader("Mean Absolute Error of the model is:")
#     disp_col.write(mean_absolute_error(y_test, prediction))
#
#     disp_col.subheader("Mean Squared Error of the model is:")
#     disp_col.write(mean_squared_error(y_test, prediction))
#
#     disp_col.subheader("R Squared Score of the model is:")
#     disp_col.write(r2_score(y_test, prediction))


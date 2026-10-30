# Website for the Customer Churn Predictor
# Run (from this folder):  streamlit run app.py

import sys
import pandas as pd
import streamlit as st

sys.dont_write_bytecode = True  # don't create a __pycache__ folder when importing main.py

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

# Importing main.py runs the whole pipeline once (load, clean, explore, train, compare).
# After that we can use its variables: main.nn, main.rf, main.fig1, ...
with st.spinner("Training the models, please wait..."):
    import main

st.title("📉 Customer Churn Predictor")
st.caption("Will this customer leave (churn) or stay?")

# ---- Sidebar: the inputs ----
st.sidebar.header("Enter the customer details")
age = st.sidebar.slider("Age", 18, 80, 40)
gender = st.sidebar.selectbox("Gender", ["Female", "Male", "Other"])
tenure = st.sidebar.slider("Tenure (months with us)", 1, 72, 12)
monthly = st.sidebar.slider("Monthly charges", 10.0, 150.0, 70.0)
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.sidebar.selectbox(
    "Payment method", ["Bank transfer", "Credit card", "Electronic check", "Mailed check"])

# Total charges is roughly tenure x monthly charges, so we work it out instead of asking
total = round(tenure * monthly, 2)
st.sidebar.caption(f"Total charges (tenure × monthly): {total}")

predict_tab, explore_tab, compare_tab = st.tabs(["🔮 Predict", "📊 Explore data", "🏆 Compare models"])

# ---- Tab 1: prediction ----
with predict_tab:
    st.subheader("Chance that this customer churns")
    chances = main.predict_churn(age, gender, tenure, monthly, contract, payment, total)

    cols = st.columns(3)
    for col, (name, p) in zip(cols, chances.items()):
        col.metric(name, f"{p * 100:.1f}%")

    best = chances[main.winner]
    if best >= 0.5:
        st.error(f"{main.winner} says: HIGH risk of leaving ({best * 100:.1f}%)")
    else:
        st.success(f"{main.winner} says: likely to stay (churn chance {best * 100:.1f}%)")
    st.info(f"The {main.winner} scored best on the test data, so trust its number more.")

# ---- Tab 2: plots ----
with explore_tab:
    plots = {
        "Churn by contract": main.fig1,
        "Churn by tenure": main.fig2,
        "Churn by monthly charges": main.fig3,
    }
    choice = st.radio("Choose a plot", list(plots.keys()), horizontal=True)
    st.pyplot(plots[choice])

# ---- Tab 3: model comparison ----
with compare_tab:
    st.table(main.results)
    st.success(f"Winner (highest ROC-AUC): {main.winner}")
    st.pyplot(main.fig4)

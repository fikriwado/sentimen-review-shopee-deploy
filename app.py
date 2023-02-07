import joblib
import numpy as np
import streamlit as st

# Vect and Model path
vect_path = "./vectorizer.pkl"
model_path = "./model.pkl"

# Load model and vectorizer
vect = joblib.load(vect_path)
model = joblib.load(model_path)

# Predict text
def prediction(text):
    text = np.array([text], dtype=object)
    sample = vect.transform(text)
    predicted = model.predict(sample)
    return predicted

html_temp = """
<div style ="background-color:#eece90;padding:13px;">
    <h1 style ="color:black;text-align:center;">Sentiments Analysis Shopee Review</h1>
</div>
<div style ="background-color:#ffd98e; border-top: solid 1px #cca558; margin-bottom: 24px;">
    <h2 style ="color:black;text-align:center;">Fikri & Sahrul</h2>
</div>
"""

st.markdown(html_temp, unsafe_allow_html = True)

review = st.text_input("Review", "Pengalaman belanja yang baik")
result =""
	
if st.button("Predict"):
    result = prediction(review)[0]

if result != "":
    st.success('Review yang dimasukan: {}'.format(result))

import streamlit as st
import streamlit_book as stb
from nltk import sent_tokenize

st.set_page_config()
st.title("MCQ Generation")
para = st.text_area("", placeholder="Insert context here...")
num_qns = st.text_input("", max_chars=2) 
result = st.button("Generate")

passage = sent_tokenize(para)

if result:
    st.write(passage[0])
    st.write("Number of questions", num_qns)
    # for i in range(int(num_qns)):
    stb.single_choice(f"{1}. {passage[0]}", options=["A", "B", "C", "D"], answer_index=1)
from secrets import choice
from turtle import onclick
import streamlit as st
import streamlit_book as stb
from streamlit_tags import st_tags
import streamlit.components.v1 as components
from nltk import sent_tokenize
from copy import deepcopy
import pandas as pd
import time

# Read CSS to style the UI
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

placeholder = st.empty()
uploaded_file = placeholder.file_uploader("Choose a file", type=["csv"])

if uploaded_file:

    # Remove the upload button
    placeholder.empty()
    filename = uploaded_file.name[:-4]
    st.title(filename)
    df = pd.read_csv(uploaded_file)

    choice_ls = df[["choice1", "choice2", "choice3", "choice4"]].values.tolist()
    ans_ls = df[["answer"]].values.tolist()

    st.session_state.clear()    
    if "submitted" not in st.session_state:
        st.session_state["submitted"] = False

    ################################
    # Front-end to show questions 
    ################################

    chosen_ans_ls = []
    for i in range(len(ans_ls)):        
        # Display text inputs with anwers from AI model as default
        st.write(f"**Question {i}**")
        answer = st.radio("", (choice_ls[i][0], choice_ls[i][1], choice_ls[i][2], choice_ls[i][3]))
        chosen_ans_ls.insert(i, answer)
    
    total_score = 0
    for i, answer in enumerate(chosen_ans_ls):
        if answer == ans_ls[i][0]:
            total_score+=1

    placeholder = st.empty()
    submitted = placeholder.button("Submit")
    if submitted:
        # Remove the submit button
        placeholder.empty()
        st.session_state.submitted = True

    if st.session_state.submitted:
        st.subheader(f"Score: {total_score/len(ans_ls)*100}")
        st.header("Answers")

        for i in range(len(ans_ls)):        
        # Display answers
            st.write(f"**Question {i}**")
            st.write(f"Answer: {ans_ls[i][0]}")

    st.session_state.clear()
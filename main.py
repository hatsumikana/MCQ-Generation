from turtle import onclick
import streamlit as st
import streamlit_book as stb
from nltk import sent_tokenize

# Read CSS to style the UI
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# st.session_state
# Initialize session state variables 
if "num_qns" not in st.session_state:
    st.session_state["num_qns"] = 0

if "generated" not in st.session_state:
    st.session_state["generated"] = False

if "disabled_count" not in st.session_state:
    st.session_state["disabled_count"] = 0

if "submitted" not in st.session_state:
    st.session_state["submitted"] = False

################################
# Front-end to accept inputs
################################
st.title("MCQ Generation")
topic = st.text_input("", placeholder="Insert topic here...") 
para = st.text_area("", placeholder="Insert context here...")
num_qns = st.text_input("", max_chars=2) 
generated = st.button("Generate questions")

################################
# AI pipeline
################################
passage = sent_tokenize(para)

choice_ls = [["A", "B", "C", "D"],
            ["B", "C", "C", "D"],
            ["E", "F", "C", "D"],
            ["B", "G", "C", "D"]]

ans_ls = [ "B", "C", "E", "G"]

################################
# Front-end to show questions based on AI pipeline 
################################

# Update session state variables
if num_qns:
    st.session_state["num_qns"] = int(num_qns)

if generated:
    st.session_state["generated"] = generated
    for i in range(int(num_qns)):
        for j in range(1,5):
            if f"{i}{j}_disabled" not in st.session_state:
                st.session_state[f"{i}{j}_disabled"] = False
            if f"{i}5" not in st.session_state:
                st.session_state[f"{i}5_disabled"] = False

def ans_cb(choice1, choice2, choice3, choice4, button):
    st.session_state[choice1] = True
    st.session_state[choice2] = True
    st.session_state[choice3] = True
    st.session_state[choice4] = True
    st.session_state[button] = True
    st.session_state.disabled_count +=4

if st.session_state["generated"]:

    for i in range(int(num_qns)):            
        st.write(f"{i+1}. {passage[i]}")
        st.text_input("",value=choice_ls[i][0], key=f"{i}1", disabled=st.session_state[f"{i}1_disabled"]) 
        st.text_input("",value=choice_ls[i][1], key=f"{i}2", disabled=st.session_state[f"{i}2_disabled"]) 
        st.text_input("",value=choice_ls[i][2], key=f"{i}3", disabled=st.session_state[f"{i}3_disabled"]) 
        st.text_input("",value=choice_ls[i][3], key=f"{i}4", disabled=st.session_state[f"{i}4_disabled"])
        ans = st.button("Accept Answers", key=f"{i}5", on_click=ans_cb, args=(f"{i}1_disabled", f"{i}2_disabled", f"{i}3_disabled", f"{i}4_disabled", f"{i}5_disabled"), disabled=st.session_state[f"{i}5_disabled"])

if st.session_state.disabled_count//4 == st.session_state.num_qns and st.session_state.disabled_count>0 and st.session_state.num_qns>0:
    submitted = st.button("Submit")
    if submitted:
        st.session_state.submitted = True

if st.session_state.submitted:
    st.write("MCQ quiz has been generated!")

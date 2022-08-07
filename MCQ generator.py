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

if "submitted_disabled" not in st.session_state:
    st.session_state["submitted_disabled"] = False

################################
# Front-end to accept inputs
################################
st.title("MCQ Generation")
st.write("**Enter Topics:**")
topics = st_tags(
    label="",
    text='Press enter to add more',
    value=['Artificial Intelligence'],
    suggestions=['neural networks', 'deep learning'],
    maxtags = 5,
    key='1')
st.write("**Enter Quiz Title:**")
quiz_title = st.text_input("", placeholder="Please enter the quiz title...") 
para = st.text_area("", placeholder="Insert context here...")
st.write("**Enter Number Of Questions:**")
num_qns = st.text_input("", max_chars=2, placeholder="Key in a number from 1-99") 
generated = st.button("Generate questions")

################################
# AI pipeline
################################

passage = sent_tokenize(para)

choice_ls = [["A", "B", "C", "D"],
            ["B", "Z", "C", "D"],
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
    with st.spinner('Generating quiz...'):
        time.sleep(5)

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

# define border colors for correct and wrong
wrong = "lightcoral"
correct = "lightgreen"

if st.session_state["generated"]:

    for i in range(int(num_qns)):        
        # Display text inputs with anwers from AI model as default    
        st.write(f"{i+1}. {passage[i]}")
        st.text_input("",value=choice_ls[i][0], key=f"{i}1", disabled=st.session_state[f"{i}1_disabled"]) 
        st.text_input("",value=choice_ls[i][1], key=f"{i}2", disabled=st.session_state[f"{i}2_disabled"]) 
        st.text_input("",value=choice_ls[i][2], key=f"{i}3", disabled=st.session_state[f"{i}3_disabled"]) 
        st.text_input("",value=choice_ls[i][3], key=f"{i}4", disabled=st.session_state[f"{i}4_disabled"])
        
        # Get index of correct answer
        idx_correct = choice_ls[i].index(ans_ls[i])
        
        # Edit border colors 
        components.html(
            f"""
            <script>
            const elements = window.parent.document.querySelectorAll('.stTextInput div[data-baseweb="input"] > div')
            elements[{i*4+2}].style.border = 'medium solid {correct if (idx_correct+(i*4)+1) == (i*4+1) else wrong}'
            elements[{i*4+2}].style.borderRadius = '4px'
            elements[{i*4+3}].style.border = 'medium solid {correct if (idx_correct+(i*4)+1) == (i*4+2) else wrong}'
            elements[{i*4+3}].style.borderRadius = '4px'
            elements[{i*4+4}].style.border = 'medium solid {correct if (idx_correct+(i*4)+1) == (i*4+3) else wrong}'
            elements[{i*4+4}].style.borderRadius = '4px'
            elements[{i*4+5}].style.border = 'medium solid {correct if (idx_correct+(i*4)+1) == (i*4+4) else wrong}'
            elements[{i*4+5}].style.borderRadius = '4px'
            </script>
            """,
            height=0,
            width=0,
        )
        ans = st.button("Accept Answers", key=f"{i}5", on_click=ans_cb, args=(f"{i}1_disabled", f"{i}2_disabled", f"{i}3_disabled", f"{i}4_disabled", f"{i}5_disabled"), disabled=st.session_state[f"{i}5_disabled"])

def submit_cb():
    st.session_state.submitted_disabled = True

if st.session_state.disabled_count//4 == st.session_state.num_qns and st.session_state.disabled_count>0 and st.session_state.num_qns>0:
    submitted = st.button("Submit", on_click= submit_cb, disabled=st.session_state.submitted_disabled)
    if submitted:
        st.session_state.submitted = True

def save_to_excel():
    combined_ls = []
    for i, ls in enumerate(choice_ls):
        temp_ls = deepcopy(ls)
        temp_ls.append(ans_ls[i])
        combined_ls.append(temp_ls)
    df = pd.DataFrame(combined_ls, columns=["choice1", "choice2", "choice3", "choice4", "answer"])
    df.to_csv(f"{quiz_title}.csv", index=False)

if st.session_state.submitted:
    save_to_excel()
    st.subheader("MCQ quiz has been generated! Go to quiz page to review quiz")
    st.session_state.clear()
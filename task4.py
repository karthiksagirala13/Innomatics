import streamlit as st
import google.generativeai as genai
from IPython.display import display, Markdown



key = 'AIzaSyBvCQeOuEJRW5i75HzyW6X_xJIh79ktez8'
genai.configure(api_key=key)
system_prompt = """You are a Python code reviewer. Identify bugs and provide fixed code. you should analyze the submitted code and identify potential bugs, errors, or areas of improvement.
you should also provide the fixed code snippets.
"""
model = genai.GenerativeModel(model_name='models/gemini-2.0-flash-exp',
                              system_instruction=system_prompt)
def review_code(user_code):
    response = model.generate_content(user_code)
    return response.text

# Streamlit UI
st.title("📢 An AI Code Reviewer")
st.write("Enter your Python code here ...")

user_code = st.text_area("Python Code:")

if st.button("Generate") and user_code:
    with st.spinner("Analyzing..."):
        feedback = review_code(user_code)
    
    
    st.markdown(feedback, unsafe_allow_html=True)

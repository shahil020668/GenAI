from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

import streamlit as st

load_dotenv()

st.header("Research Tool")

# model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

user_input = st.text_input('enter prompt')

if st.button('Summarize'):
    result = model.invoke(user_input)
    st.write(result.content)

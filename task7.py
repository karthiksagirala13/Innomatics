import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Streamlit page config
st.set_page_config(page_title="Data Science Chatbot", layout="wide")

# Initialize chat history in session state (this must come before any references to it)
if "messages" not in st.session_state:
    # Start with a system message
    st.session_state.messages = [
        {"role": "system", "content": "You are a friendly and helpful data science tutor. When users share their name, remember it and use it in your responses."}
    ]

# Initialize API Key
api_key = key  # Replace with your actual API key or use secrets

# Initialize Chat Model
chat_model = ChatGoogleGenerativeAI(google_api_key=api_key, model='models/gemini-2.0-flash-exp')

# Streamlit UI
st.title("🤖 Conversational AI Data Science Tutor")
st.write("Ask me any Data Science-related question!")

# Display chat history (skipping the system message)
for message in st.session_state.messages:
    if message["role"] != "system":
        st.chat_message(message["role"]).write(message["content"])

# Chat Interface
if user_input := st.chat_input("Type your Data Science question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    st.chat_message("user").write(user_input)
    
    # Prepare messages for the API call
    messages_for_api = [
        SystemMessage(content=st.session_state.messages[0]["content"])
    ]
    
    for msg in st.session_state.messages[1:]:
        if msg["role"] == "user":
            messages_for_api.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages_for_api.append(AIMessage(content=msg["content"]))
    
    # Get AI response
    response = chat_model.invoke(messages_for_api)
    
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    
    # Display AI response
    st.chat_message("assistant").write(response.content)

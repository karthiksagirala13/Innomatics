import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser


# Secure API key handling
api_key = 'AIzaSyBvCQeOuEJRW5i75HzyW6X_xJIh79ktez8'  # Store in .streamlit/secrets.toml

# System prompt for travel recommendations

prompt_template = ChatPromptTemplate.from_messages([
    ("system","""You are an AI travel planner. Given a source and destination, 
suggest optimal travel options including cab, bus, train, and flights. 
Estimate travel time and cost based on common travel patterns.
Provide results in a structured format with travel mode, estimated time, and price.
"""
),
    ("human","Plan travel from {source} to {destination}")])

system_prompt = """You are an AI travel planner. Given a source and destination, 
suggest optimal travel options including cab, bus, train, and flights. 
Estimate travel time and cost based on common travel patterns.
Provide results in a structured format with travel mode, estimated time, and price.
"""

chat_model = ChatGoogleGenerativeAI(google_api_key=api_key,model='models/gemini-2.0-flash-exp')

parser = StrOutputParser()

chain = prompt_template | chat_model | parser

def get_travel_recommendations(source, destination):
    """Generates AI-powered travel recommendations."""
    user_input = {"source":source, "destination":destination}
    try:
        response = chain.invoke(user_input)
        return response if response else "No recommendations available."
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("🌍 AI-Powered Travel Planner")
st.write("Enter your source and destination to get travel recommendations.")

col1, col2 = st.columns([1, 1])  # Creates two compact columns

with col1:
    source = st.text_input("Source", placeholder="Enter city or airport")

with col2:
    destination = st.text_input("Destination", placeholder="Enter city or airport")

if st.button("Find Travel Options") and source and destination:
    with st.spinner("Fetching recommendations..."):
        travel_info = get_travel_recommendations(source, destination)

    st.subheader("Travel Recommendations")
    st.markdown(travel_info, unsafe_allow_html=True)

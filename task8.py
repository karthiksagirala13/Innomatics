import streamlit as st
import pandas as pd
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import google.generativeai as genai
import speech_recognition as sr
import tempfile
from pydub import AudioSegment

# Streamlit UI
st.title("🤖 AI-Powered Subtitle Search Chatbot")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    if message['role'] == 'user':
        with st.chat_message("user"):
            st.write(message['content'])
    else:
        with st.chat_message("assistant"):
            st.write(message['content'])

# Load CSV directly from the repository
csv_file_path = "pages/dbdata.csv"  # Ensure this file is in your repo
if os.path.exists(csv_file_path):
    df = pd.read_csv(csv_file_path)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=50)
    documents = [Document(page_content=text) for text in df['file_content'].tolist()]
    chunks = text_splitter.split_documents(documents)
    
    # Set API key
    os.environ["GOOGLE_API_KEY"] = key
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    
    # Create embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Initialize ChromaDB
    persist_directory = "./chroma_db"
    db = Chroma(collection_name="vector_database", embedding_function=embedding_model, persist_directory=persist_directory)
    
    # Store embeddings in ChromaDB
    db.add_documents(chunks)
    
    # Audio-to-text function with MP3 support
    def audio_to_text(audio_bytes, file_extension):
        r = sr.Recognizer()
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_input:
            temp_input.write(audio_bytes)
            temp_input_name = temp_input.name
        
        # Convert to WAV if MP3
        if file_extension.lower() == '.mp3':
            wav_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sound = AudioSegment.from_mp3(temp_input_name)
            sound.export(wav_file.name, format="wav")
            audio_file_path = wav_file.name
        else:
            audio_file_path = temp_input_name
        
        # Process audio
        try:
            with sr.AudioFile(audio_file_path) as source:
                audio_data = r.record(source)
                text = r.recognize_google(audio_data)
                return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand the audio"
        except sr.RequestError:
            return "Sorry, speech service is unavailable"
        except Exception as e:
            return f"Error processing audio: {str(e)}"
        finally:
            # Clean up temporary files
            os.unlink(temp_input_name)
            if file_extension.lower() == '.mp3':
                os.unlink(audio_file_path)
    
    # Audio input
    st.write("**Upload audio to search by voice:**")
    audio_input = st.file_uploader("Upload audio", type=["wav", "mp3"])
    
    # Text input
    user_input = st.chat_input("Ask me about subtitles...")
    
    # Process audio if provided
    if audio_input is not None:
        with st.spinner("Converting audio to text..."):
            audio_bytes = audio_input.read()
            file_extension = os.path.splitext(audio_input.name)[1]
            text_from_audio = audio_to_text(audio_bytes, file_extension)
            
            if not text_from_audio.startswith(("Sorry", "Error")):
                user_input = text_from_audio
                st.info(f"Recognized text: {user_input}")
            else:
                st.error(text_from_audio)
    
    if user_input:
        # Add user message to chat history and display it
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Process the query
        with st.chat_message("assistant"):
            with st.spinner("Searching subtitles..."):
                docs_chroma = db.similarity_search_with_score(user_input, k=3)
                
                if docs_chroma:
                    context_text = "\n\n".join([doc.page_content for doc, score in docs_chroma])
                    
                    PROMPT_TEMPLATE = """
                    Answer the question based only on the following subtitle context:
                    {context}
                    
                    Question: {question}
                    Provide a concise yet informative response.
                    """
                    
                    prompt = PROMPT_TEMPLATE.format(context=context_text, question=user_input)
                    model = genai.GenerativeModel(model_name='models/gemini-2.0-flash-exp')
                    response_text = model.generate_content(prompt)
                    
                    st.write(response_text.text)
                    # Add AI response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text.text})
                else:
                    st.write("I couldn't find any relevant information in the subtitles.")
                    # Add AI response for no results
                    st.session_state.chat_history.append({"role": "assistant", "content": "I couldn't find any relevant information in the subtitles."})
else:
    st.error("❌ CSV file not found! Please ensure 'dbdata.csv' is present in the repository.")

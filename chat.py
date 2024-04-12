import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import google.generativeai as genai
import os
import base64

# Set up Google Generative AI API key
genai.configure(api_key="AIzaSyCR8MnI9MejtOYsSMljNbBqlZRHWfbxWl0")
os.environ['GOOGLE_API_KEY'] = 'AIzaSyCR8MnI9MejtOYsSMljNbBqlZRHWfbxWl0'

# Initialize the Generative AI model
my_llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)

# Set Streamlit page configuration
st.set_page_config(page_title="Character AI", page_icon="✌️", layout="centered")

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)
    
set_background('wallpaperflare.com_wallpaper.jpg')
    
st.header("Character AI")
st.subheader("Chat with your character")


def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


with st.sidebar:
    st.title("Choose the Character")
    character_name = st.text_input("Character name")
    st.title("Choose the Series Name")
    series_name = st.text_input("Series name")

    # Create the prompt template
    my_prompt = PromptTemplate.from_template(
        "you are {character_name}, a fictional character from {series_name}, and you have the personality of {character_name}. Answer as if you are the character being asked by the fan: {question}"
    )


# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["message"])    
# Chat input and sidebar for character and series selection
user_prompt = st.text_input("Ask me")

if user_prompt:
    
    st.chat_message("user").markdown(user_prompt)
    # Initialize LLM Chain
    chain = LLMChain(
        llm=my_llm,
        prompt=my_prompt
    )

    # Pass input data to LLM Chain
    input_data = {
        'character_name': character_name,
        'series_name': series_name,
        'question': user_prompt
    }

    # Generate response
    response = chain.invoke(input=input_data)
    text_result = response["text"]
    
    # Display Gemini-Pro's response
    with st.chat_message("assistant"):
        st.markdown(text_result)
    
    # Add user and assistant messages to chat history
    st.session_state.chat_history.append({"role": "user", "message": user_prompt})
    st.session_state.chat_history.append({"role": "assistant", "message": text_result})    

    
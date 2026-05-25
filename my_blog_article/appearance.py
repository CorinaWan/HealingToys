import streamlit as st
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# Page Configuration
st.set_page_config (
    page_title="AI Counsellor - DeepSeek AI Model",
    page_icon="🤖",
    layout="centered",
)

# CSS Design for UI
st.markdown("""
<style> 
.chat-container {
    background-color: #0f172a;
    padding: 20px;
}
.user-msg {
    background-color: #22563eb;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    text-align: right;
}
.ai-msg {
    background-color: #1e293b;
    color: white;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    text-align: left;
}
.title {
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    color: #FFD25D;
}
.subtitle {
    text-align: center;
    color: orange;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Title and Subtitle
st.markdown('<div class="title">AI Counsellor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by DeepSeek AI Model, which is your confidant</div>', unsafe_allow_html=True)

# SideBar for User Input
st.sidebar.title("Settings")

modal_name = st.sidebar.selectbox(
    "Select AI Model",
    [
        "deepseek-r1:1.5b", 
    ]
)

if st.sidebar.button("Reset Conversation"):
    st.session_state.chat = []
    st.rerun()
    
st.sidebar.markdown("___")
st.sidebar.write("AI Counsellor is your best confidant, always ready to listen and provide mental support. ")

# Session State Initialization
if "chat" not in st.session_state:
    st.session_state.chat = []



#Prompt 
template = """You are a helpful and passionate AI Counsellor, designed to provide emotional mental support to students.

Question: {question} 

Answer:"""

prompt = ChatPromptTemplate.from_template(template)
model = Ollama(model=modal_name)
chain = prompt | model
                 
# Chat DISPLAY
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for chat in st.session_state.chat:
    if chat["role"] == "user":
         st.markdown(f'<div class="user-msg">{chat["content"]}</div>', unsafe_allow_html=True)
    else:
         st.markdown(f'<div class="ai-msg">{chat["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# input
question = st.text_input("Type your message here...", placeholder="Ask anything about mental health, stress management, or just need someone to talk to.")

# Typing Effect Function
def typing_effect(text, speed=0.02):
    placeholder = st.empty()
    typed_text = ""
    for char in text:
        typed_text += char
        placeholder.markdown(f'<div class="ai-msg">{typed_text}</div>', unsafe_allow_html=True)
        time.sleep(speed)

#Send Button
if st.button("Send"):
    if question:
        st.session_state.chat.append({"role":"user", "content": question})
        
        with st.spinner("AI Counsellor is thinking..."):
           response = chain.invoke({"question":question})
           
        # Typing Animation
        typing_effect(response)

        st.session_state.chat.append({"role":"ai", "content": response})
        st.rerun()
    else:
        st.warning("Please enter a message before sending.")
        
# Reference website: https://www.youtube.com/watch?v=qwo6AT9rOqQ
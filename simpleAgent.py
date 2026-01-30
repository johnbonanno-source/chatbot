import streamlit as st 
st.set_page_config(page_title="Chatbot", page_icon="💬")

from collections import deque

import typing
import os
from langchain_ollama import ChatOllama


def get_llm() -> ChatOllama:
    llm = ChatOllama(
        model="gemini-3-flash-preview",             # Model identifier.
        base_url="https://ollama.com",              # Ollama-compatible API endpoint.
        api_key=os.environ.get("OLLAMA_API_KEY"),   # Read API key from the environment.
    )
    return llm

def get_summarizer() -> ChatOllama:
    llm = ChatOllama(
        model="kimi-k2.5:cloud",                    # Model identifier.
        base_url="https://ollama.com",              # Ollama-compatible API endpoint.
        api_key=os.environ.get("OLLAMA_API_KEY"),   # Read API key from the environment.
    )
    return llm


def main(system_prompt:str, history:deque):

    st.title("Chatbot")

    # set up the data structure for history
    if "history" not in st.session_state:
        st.session_state.history = deque(maxlen=20)
    # pass in the system prompt
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = system_prompt

    if len(st.session_state.history) == 4:
        summarizeHistory()
        
    for role, text in st.session_state.history:
        st.chat_message(role).write(text)


    user_text = st.chat_input("")
    if not user_text:
        return
    
    st.session_state.history.append(("user", user_text))
    st.chat_message("user").write(user_text)

    context_prompt = [("system", st.session_state.system_prompt)] + list(st.session_state.history)
    llm = get_llm()
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            resp = llm.invoke(context_prompt)
        st.write(resp.content)
    st.session_state.history.append(("assistant", resp.content))

def summarizeHistory():
    print("summarizing hist")
    history = st.session_state.history
    lines = []
    while len(history) > 5:
        role,text = history.popleft()
        print(role,text)
        lines.append(f"{role}: {text}")
    prompt = [
        ("system", "Update the running conversation summary. Return ONLY the updated summary."),
        ("user", "".join(lines)),
    ]
    st.session_state.history.append(("assistant",get_summarizer().invoke(prompt).content.strip()))
    print(st.session_state.history)

if __name__ == "__main__":

    system_prompt = "You are a concise assistant. You follow up every response with a follow on question, on a new line, similar to the following, but create variation ex: Would you like to know more about a specific topic, or anything else? I will be here, boss/cheif/king"
    d = deque()
    main(system_prompt,d)

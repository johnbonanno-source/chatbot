import yfinance as yf
import os
import re
import json
import inspect
import pandas as pd
from functools import lru_cache
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

from prompts import SYSTEM_PROMPT, EXTRACT_ACTION_AND_TICKER_PROMPT, EXTRACT_RELEVANT_METHOD_PROMPT

@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",             # Model identifier.
        api_key=os.environ.get("GEMINI_API_KEY"),   # Read API key from the environment.
    )
    return llm

def getMethodsFromticker(ticker: str) -> list:
    """Using the YFI api, check what methods can be called on a given ticker"""
    t = yf.Ticker(ticker)
    methods = []
    for name, member in inspect.getmembers(type(t)):
        if name.startswith("_"):
            continue
        if callable(member):
            methods.append(name)
    return sorted(m for m in methods)

def get_ticker_and_action_from_query(user_text:str)->list:
    """Extract a stock ticker and a action the user would like performed, based on their input query"""
    llm = get_llm()
    prompt = [
        ("system", EXTRACT_ACTION_AND_TICKER_PROMPT),
        ("user", user_text),
    ]

    resp = llm.invoke(prompt)
    content = json.loads(resp.text.strip())  # Get the text content
    ticker = content.get("ticker")
    action = content.get("action")
    return (ticker, action)

    

def get_specialized_methods_from_llm(action: str, all_methods:list)->list:
    """Of all methods callable on a specific stock ticker, return which of these methods relate to the action requested by the user, in a list format"""
    llm = get_llm()
    
    prompt = [
        ("system", EXTRACT_RELEVANT_METHOD_PROMPT),
        ("user", f"Action: {action}\nAllowed methods: {', '.join(all_methods)}"),
    ]
    
    resp = llm.invoke(prompt)
    content = json.loads(resp.text)
    
    if not content:
        return []
    else:
        return content

def yahoo_finance(ticker_symbol:str, method_list:list) -> dict:
    """For each method in method list, call the method and store in a dictionary defined as methodName:methodOutput"""
    output = dict()
    if ticker_symbol != 'None':
        ticker = yf.Ticker(ticker_symbol)
        for method_name in method_list:
            try:
                if method_name.startswith("history"):
                    output["history"] = eval(f"ticker.{method_name}")
                else:
                    method = getattr(ticker, method_name, None)
                    if callable(method):
                        output[method_name] = method()
            except Exception as e:
                    output[method_name] = f"Error: {e}"
    return output

def display_stock_chart(ticker: str, yfi_output: dict) -> None:
    """Generate a stock chart based on historical data"""
    import streamlit as st
    if "history" in yfi_output and not yfi_output["history"].empty:
        history_df = yfi_output.get("history")
        if history_df is not None and not history_df.empty:
            close = history_df["Close"]
            if close.empty:
                return
            ymin = float(close.min())
            ymax = float(close.max())
            yrange = ymax - ymin
            pad = max(yrange * 0.02, ymax * 0.001, 1e-6)
            y_domain = [ymin - pad, ymax + pad]
            df = history_df.reset_index()
            time_col = "Date" if "Date" in df.columns else "Datetime"  # yfinance uses one of these
            spec = {
                "mark": {"type": "line"},
                "encoding": {
                    "x": {"field": time_col, "type": "temporal"},
                    "y": {
                    "field": "Close",
                    "type": "quantitative",
                    "scale": {"domain": y_domain},
                    },
                },
            }
            st.subheader(f"{ticker} - Stock Performance")
            st.vega_lite_chart(df[[time_col, "Close"]], spec, width="stretch")

def generate_final_response(history: list, yfi_output: dict) -> str:
    """Generate final LLM response with Yahoo Finance context."""
    llm = get_llm()
    messages = [("system", SYSTEM_PROMPT)] + history
    if yfi_output:
        messages.append(("system", f"Yahoo Finance tool output (JSON):\n{yfi_output}"))
    resp = llm.invoke(messages)
    return re.sub(r'\*+', '', resp.text).strip()

def summarizeHistory(history: list) -> list:
    """Trim conversation history by summarizing older messages."""
    N = len(history)
    toSummarize = history[:N-5]
    remaining = history[N-5:]
    chunk = "\n".join(f"{role}: {text}" for role, text in toSummarize)
    prompt = [
        ("system", "Update the running conversation summary. Return ONLY the updated summary."),
        ("user", chunk),
    ]
    remaining.append(("assistant",get_llm().invoke(prompt).text.strip()))
    return remaining

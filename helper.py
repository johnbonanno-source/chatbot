import yfinance as yf
import os
import re
import json
import pandas as pd
from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

from prompts import methods, SYSTEM_PROMPT, EXTRACT_ACTION_AND_TICKER_PROMPT, EXTRACT_RELEVANT_METHOD_PROMPT

@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    print("[DEBUG] get_llm called")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",             # Model identifier.
        api_key=os.environ.get("GEMINI_API_KEY"),   # Read API key from the environment.
    )
    return llm

def get_ticker_and_action_from_query(user_text: str) -> dict:
    """Extract stock request plan from user query."""
    print(f"[DEBUG] get_ticker_and_action_from_query called | user_text={user_text!r}")
    llm = get_llm()
    prompt = [("system", EXTRACT_ACTION_AND_TICKER_PROMPT), ("user", user_text)]
    resp = llm.invoke(prompt)
    try:
        content = json.loads(resp.text.strip())
    except json.JSONDecodeError:
        return {"ticker": None, "action": user_text, "methods": [], "history": {"period": "5d"}}
    if not isinstance(content, dict):
        return {"ticker": None, "action": user_text, "methods": [], "history": {"period": "5d"}}

    content.setdefault("ticker", None)
    content.setdefault("action", user_text)
    content.setdefault("methods", [])
    content.setdefault("history", {})

    if not isinstance(content["methods"], list):
        content["methods"] = []
    if not isinstance(content["history"], dict):
        content["history"] = {}
    content["history"].setdefault("period", "5d")
    return content


def get_specialized_methods_from_llm(action: str, all_methods:list)->list:
    """Of all methods callable on a specific stock ticker, return which of these methods relate to the action requested by the user, in a list format"""
    print(f"[DEBUG] get_specialized_methods_from_llm called | action={action!r}, all_methods_count={len(all_methods)}")
    llm = get_llm()
    prompt = [("system", EXTRACT_RELEVANT_METHOD_PROMPT), ("user", f"Action: {action}\nAllowed methods: {', '.join(all_methods)}")]    
    resp = llm.invoke(prompt)
    content = json.loads(resp.text)
    return content if content else []

def choose_interval(period: str) -> str:
    p = (period or "5d").lower().strip()
    # Yahoo intraday intervals are only available for limited lookback windows.
    if p.endswith("y") or p.endswith("mo"):
        return "1d"
    return "1h"

def yahoo_finance(ticker_symbol: str, method_list: list, history_cfg: dict | None = None) -> dict:
    """For each method in method list, call the method and store in a dictionary defined as methodName:methodOutput"""
    print(
        f"[DEBUG] yahoo_finance called | ticker_symbol={ticker_symbol!r}, "
        f"method_list={method_list!r}, history_cfg={history_cfg!r}"
    )
    output = dict()
    history_cfg = history_cfg or {}
    history_period = history_cfg.get("period", "5d")
    history_interval = choose_interval(history_period)

    if ticker_symbol and ticker_symbol is not None:
        ticker = yf.Ticker(ticker_symbol)
        for method_name in method_list:
            try:
                if method_name.startswith("history"):
                    output["history"] = ticker.history(period=history_period, interval=history_interval).tail(50)
                    continue

                if method_name == "live" or method_name.startswith("live("):
                    output["live"] = "Skipped: streaming method disabled"
                    continue

                if method_name not in methods:
                    output[method_name] = "Skipped: Disallowed Method"
                    continue

                method = getattr(ticker, method_name, None)
                if callable(method):
                    output[method_name] = method()
                else:
                    output[method_name] = "Skipped: Not callable"
            except Exception as e:
                output[method_name] = f"Error: {e}"
    return output

def display_stock_chart(ticker: str, yfi_output: dict) -> None:
    """Render stock close price over time (x=date, y=price)."""
    print(f"[DEBUG] display_stock_chart called | ticker={ticker!r}, yfi_output_type={type(yfi_output).__name__}")

    history_df = yfi_output.get("history") if isinstance(yfi_output, dict) else None
    if history_df is None or not hasattr(history_df, "columns") or "Close" not in history_df.columns:
        return

    df = history_df.reset_index()
    time_col = "Date" if "Date" in df.columns else "Datetime"
    if time_col not in df.columns:
        return

    plot_df = df[[time_col, "Close"]].copy()
    plot_df[time_col] = pd.to_datetime(plot_df[time_col], errors="coerce")
    plot_df["Close"] = pd.to_numeric(plot_df["Close"], errors="coerce")
    plot_df = plot_df.dropna(subset=[time_col, "Close"]).sort_values(time_col)
    if plot_df.empty:
        return

    close = plot_df["Close"]
    ymin = float(close.min())
    ymax = float(close.max())
    pad = max((ymax - ymin) * 0.04, 1e-6)

    spec = {
        "width": 920,
        "height": 460,
        "layer": [
            {
                "mark": {"type": "line", "strokeWidth": 2.2, "color": "#0f766e"},
                "encoding": {
                    "x": {
                        "field": time_col,
                        "type": "temporal",
                        "axis": {
                            "title": "Date",
                            "grid": False,
                            "labelAngle": -25,
                            "tickCount": 8,
                            "format": "%Y-%m-%d %H:%M",
                        },
                    },
                    "y": {
                        "field": "Close",
                        "type": "quantitative",
                        "scale": {"domain": [ymin - pad, ymax + pad], "nice": False},
                        "axis": {"title": "Close", "grid": True, "gridOpacity": 0.15, "tickCount": 6},
                    },
                },
            },
            {
                "mark": {"type": "point", "size": 16, "filled": True, "color": "#0f766e", "opacity": 0.45},
                "encoding": {
                    "x": {"field": time_col, "type": "temporal"},
                    "y": {"field": "Close", "type": "quantitative"},
                },
            },
        ],
    }

    st.subheader(f"{ticker} - Stock Performance")
    st.vega_lite_chart(plot_df, spec, use_container_width=True)

def generate_final_response(history: list, yfi_output: dict) -> str:
    """Generate final LLM response with Yahoo Finance context."""
    print(f"[DEBUG] generate_final_response called | history_len={len(history)}, has_yfi_output={bool(yfi_output)}")
    llm = get_llm()
    messages = [("system", SYSTEM_PROMPT)] + history
    if yfi_output:
        messages.append(("system", f"Yahoo Finance tool output (JSON):\n{yfi_output}"))
    resp = llm.invoke(messages)
    return re.sub(r'\*+', '', resp.text).strip()

def summarizeHistory(history: list) -> list:
    """Trim conversation history by summarizing older messages."""
    print(f"[DEBUG] summarizeHistory called | history_len={len(history)}")
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

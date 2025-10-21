"""Streamlit front-end for the EduPolicy Agent."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import requests
import streamlit as st

CONFIG_PATH = Path(__file__).with_name("openwebui_config.json")
if CONFIG_PATH.exists():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
else:
    config = {"backend_url": "http://localhost:8000", "title": "EduPolicy Agent", "description": ""}

API_URL = config.get("backend_url", "http://localhost:8000")

st.set_page_config(page_title=config.get("title", "EduPolicy Agent"), layout="wide")
st.title(config.get("title", "EduPolicy Agent"))
st.write(config.get("description", ""))

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 Bắt đầu phiên mới"):
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.messages = []
        st.rerun()

with col1:
    user_input = st.chat_input("Nhập câu hỏi về quy chế, sinh viên hoặc quy định...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"session_id": st.session_state.session_id, "message": user_input},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload.get("answer", "")
        reasoning = payload.get("reasoning", [])
        tools = payload.get("tool_interactions", [])
        st.session_state.messages.append({"role": "assistant", "content": answer, "reasoning": reasoning, "tools": tools})
    except requests.RequestException as exc:
        st.error(f"Không thể kết nối backend: {exc}")

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])
            if message.get("reasoning"):
                with st.expander("Hiển thị lập luận của agent"):
                    for item in message["reasoning"]:
                        st.write(item)
            if message.get("tools"):
                with st.expander("Hiển thị tương tác công cụ"):
                    for item in message["tools"]:
                        st.write(item)

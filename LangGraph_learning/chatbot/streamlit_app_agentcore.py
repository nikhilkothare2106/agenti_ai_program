import streamlit as st

from agentcore_invoke import invoke_agent, new_session_id

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# runtimeSessionId stays fixed for the life of this Streamlit session so the
# agent keeps using the same conversation thread.
if "runtime_session_id" not in st.session_state:
    st.session_state["runtime_session_id"] = new_session_id()

# loading the conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # first add the message to message_history
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    try:
        ai_message = invoke_agent(user_input, st.session_state["runtime_session_id"])
    except Exception as e:
        ai_message = f"Error calling agent: {e}"

    # first add the message to message_history
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
    with st.chat_message("assistant"):
        st.text(ai_message)
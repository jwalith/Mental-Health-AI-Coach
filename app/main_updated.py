import streamlit as st
from chatbot_logic_updated import (
    load_resources,
    load_encryption_key,
    collect_user_profile,
    initialize_session_state,
    conversation_chat,
)

st.set_page_config(page_title="Mental Health Chatbot", layout="wide")
st.title("🧠 Mental Health Support Assistant")

@st.cache_resource(show_spinner="Loading models and encoders...")
def cached_load_resources():
    return load_resources()

# Load resources
data, index, bi_encoder, cross_encoder, tokenizer, model = cached_load_resources()
cipher = load_encryption_key()
collect_user_profile(cipher)
initialize_session_state()

# User Input
user_input = st.text_input("Ask a mental health-related question:")
if user_input:
    with st.spinner("Thinking..."):
        response = conversation_chat(user_input, data, index, bi_encoder, cross_encoder, tokenizer, model, cipher)
        st.session_state["history"].append({"user": user_input, "bot": response})

# Display chat history
if st.session_state["history"]:
    st.subheader("🗨️ Conversation History")
    for i, turn in enumerate(st.session_state["history"]):
        st.markdown(f"**You:** {turn['user']}")
        st.markdown(f"**Assistant:** {turn['bot']}")
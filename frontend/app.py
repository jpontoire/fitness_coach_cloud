import streamlit as st
import requests
import os

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Fitness Coach", page_icon="💪")
st.title("Fitness Coach")

EQUIPMENT_OPTIONS = ["dumbbell", "barbell", "body weight", "cable", "band", "kettlebell", "machine"]

with st.sidebar:
    st.header("LLM Provider")
    provider = st.selectbox("Provider", ["Groq", "OpenAI"])
    api_key = st.text_input("API Key (optional, uses server default if empty)", type="password")

    st.header("Available equipment")
    selected_equipment = st.multiselect("Select what you have access to:", EQUIPMENT_OPTIONS)

def display_structured_answer(answer_text):
    lines = answer_text.split("\n")
    for line in lines:
        if line.startswith("Exercise:"):
            st.subheader(line.replace("Exercise:", "").strip())
        elif ":" in line and len(line.split(":", 1)[0].split()) <= 3:
            label, value = line.split(":", 1)
            st.markdown(f"**{label.strip()}:** {value.strip()}")
        else:
            st.write(line)

with st.form(key="question_form"):
    question = st.text_input("Ask me anything about exercises, workouts, or nutrition:")
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    if not api_key:
        st.error("Please enter your API key in the sidebar to use this app.")
    else:
        with st.spinner("Thinking..."):
            payload = {
                "question": question,
                "provider": provider.lower(),
                "api_key": api_key,
            }
            if selected_equipment:
                payload["equipment"] = selected_equipment
            try:
                response = requests.post(f"{API_URL}/ask", json=payload)
                response.raise_for_status()
                data = response.json()
                st.caption(f"Intent: {data['intent']}")
                if data["intent"] == "exercise_lookup":
                    display_structured_answer(data["answer"])
                else:
                    st.write(data["answer"])
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to the API: {e}")

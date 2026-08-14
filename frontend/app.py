import streamlit as st
import requests
import os

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Fitness Coach", page_icon="💪")
st.title("Fitness Coach")

EQUIPMENT_OPTIONS = ["dumbbell", "barbell", "body weight", "cable", "band", "kettlebell", "machine"]

with st.sidebar:
    st.header("Available equipment")
    selected_equipment = st.multiselect("Select what you have access to:", EQUIPMENT_OPTIONS)

question = st.text_input("Ask me anything about exercises, workouts, or nutrition:")

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

if st.button("Ask") and question.strip():
    with st.spinner("Thinking..."):
        try:
            payload = {"question": question}
            if selected_equipment:
                payload["equipment"] = selected_equipment

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

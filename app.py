import streamlit as st
import requests
import json



BASE_URL = "http://127.0.0.1:8001"
st.set_page_config(page_title="Suzy AI Assistant", layout="wide")
st.title("Suzy AI Assistant for English Learning")



tab1, tab2, tab3 = st.tabs(["Translation", "Grammar Check", "Test"])

"""
    Мини сайт для проекта.Написана с помощью streamlit
"""

with tab1:
    st.header("Text Translation")
    st.write("Translate text between Russian and English.")

    translate_text = st.text_area("Enter text to translate:", height=100)
    language = st.selectbox("Select translation direction:", ["English to Russian", "Russian to English"])

    if st.button("Translate"):
        if translate_text.strip():
            lang_map = {
                "English to Russian": "en_to_ru",
                "Russian to English": "ru_to_en"
            }
            payload = {"text": translate_text, "language": lang_map[language]}

            try:
                response = requests.post(f"{BASE_URL}/перевести-текст", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    st.success("Translation successful!")
                    st.write(f"**Original Text**: {result['original_text']}")
                    st.write(f"**Translated Text**: {result['translated_text']}")
                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error')}")
            except requests.RequestException as e:
                st.error(f"Failed to connect to the backend: {str(e)}")
        else:
            st.warning("Please enter text to translate.")

with tab2:
    st.header("Grammar Check")
    st.write("Check your English text for grammatical errors.")
    grammar_text = st.text_area("Enter English text to check grammar:", height=100)

    if st.button("Check Grammar"):
        if grammar_text.strip():
            payload = {"text": grammar_text}

            try:
                response = requests.post(f"{BASE_URL}/проверка-грамотности", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    st.success("Grammar check completed!")
                    st.write(f"**Result**: {result['result']}")
                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error')}")
            except requests.RequestException as e:
                st.error(f"Failed to connect to the backend: {str(e)}")
        else:
            st.warning("Please enter text to check.")

with tab3:
    st.header("English Test")
    st.write("Take a 30-question test to evaluate your English skills.")
    if "questions" not in st.session_state:
        st.session_state.questions = []
        st.session_state.current_question = 0
        st.session_state.answers = {}

    if st.button("Start Test") or st.session_state.questions:
        if not st.session_state.questions:
            try:
                response = requests.get(f"{BASE_URL}/получить-вопросы")
                if response.status_code == 200:
                    st.session_state.questions = response.json()
                    st.session_state.current_question = 0
                    st.session_state.answers = {}
                else:
                    st.error(f"Error fetching questions: {response.json().get('error', 'Unknown error')}")
            except requests.RequestException as e:
                st.error(f"Failed to connect to the backend: {str(e)}")

        if st.session_state.questions:
            question = st.session_state.questions[st.session_state.current_question]
            st.write(f"**Question {st.session_state.current_question + 1}**: {question['question']}")
            answer = st.text_input("Your answer:", key=f"q_{st.session_state.current_question}")

            if st.button("Submit Answer"):
                if answer.strip():
                    payload = {"question_id": question["id"], "answer": answer}
                    try:
                        response = requests.post(f"{BASE_URL}/проверка-ответ", json=payload)
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.answers[question["id"]] = result
                            st.success(f"Answer submitted: {result['result']}")
                            st.session_state.current_question += 1
                            if st.session_state.current_question >= len(st.session_state.questions):
                                st.write("**Test Completed!**")
                                score = sum(
                                    1 for ans in st.session_state.answers.values() if ans["result"] == "Correct")
                                st.write(f"Your score: {score}/{len(st.session_state.questions)}")
                                st.session_state.questions = []  # Reset test
                        else:
                            st.error(f"Error: {response.json().get('error', 'Unknown error')}")
                    except requests.RequestException as e:
                        st.error(f"Failed to connect to the backend: {str(e)}")
                else:
                    st.warning("Please enter an answer.")
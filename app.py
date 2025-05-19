import streamlit as st
import requests
import logging
import os
from dotenv import load_dotenv
from typing import Dict, List



load_dotenv()
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8501")



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)



st.set_page_config(page_title="Suzy AI Assistant", layout="wide")



if "questions" not in st.session_state:
    st.session_state.questions = []
    st.session_state.current_question = 0
    st.session_state.answers = {}
    st.session_state.test_started = False


def translation_tab():
    st.header("Text Translation")
    st.write("Translate text between Russian and English.")

    translate_text = st.text_area("Enter text to translate:", height=100)
    language = st.selectbox("Select translation direction", ["English to Russian", "Russian to English"])

    if st.button("Translate"):
        if translate_text.strip():
            lang_map = {
                "English to Russian": "en_to_ru",
                "Russian to English": "ru_to_en"
            }
            payload = {"text": translate_text, "language": lang_map[language]}
            try:
                response = requests.post(f"{BASE_URL}/translate", json=payload)
                response.raise_for_status()
                result = response.json()
                st.success("Translation successful!")
                st.write(f"**Original Text**: {result['original_text']}")
                st.write(f"**Translated Text**: {result['translated_text']}")
                logger.info("Translated text: %s", translate_text[:50])
            except requests.RequestException as e:
                st.error(f"Failed to connect to the backend: {str(e)}")
                logger.error("Translation error: %s", str(e))
        else:
            st.warning("Please enter text to translate.")
            logger.warning("Empty translation input")


def grammar_check_tab():
    st.header("Grammar Check")
    st.write("Check and correct English text for grammatical errors.")

    grammar_text = st.text_area("Enter English text to check grammar:", height=100)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Check Grammar"):
            if grammar_text.strip():
                payload = {"text": grammar_text}
                try:
                    response = requests.post(f"{BASE_URL}/check_grammar", json=payload)
                    response.raise_for_status()
                    result = response.json()
                    st.success("Grammar check completed!")
                    st.write(f"**Result**: {result['result']}")
                    logger.info("Grammar checked: %s", grammar_text[:50])
                except requests.RequestException as e:
                    st.error(f"Failed to connect to the backend: {str(e)}")
                    logger.error("Grammar check error: %s", str(e))
            else:
                st.warning("Please enter text to check.")
                logger.warning("Empty grammar check input")

    with col2:
        if st.button("Correct Grammar"):
            if grammar_text.strip():
                payload = {"text": grammar_text}
                try:
                    response = requests.post(f"{BASE_URL}/correct_grammar", json=payload)
                    response.raise_for_status()
                    result = response.json()
                    st.success("Grammar correction completed!")
                    st.write(f"**Corrected Text**: {result['corrected_text']}")
                    logger.info("Grammar corrected: %s", grammar_text[:50])
                except requests.RequestException as e:
                    st.error(f"Failed to connect to the backend: {str(e)}")
                    logger.error("Grammar correction error: %s", str(e))
            else:
                st.warning("Please enter text to correct.")
                logger.warning("Empty grammar correction input")


def test_tab():
    st.header("English Test")
    st.write("Take a 30-question test to evaluate your English grammar skills.")

    if st.button("Start Test") and not st.session_state.test_started:
        try:
            response = requests.get(f"{BASE_URL}/get_questions")
            response.raise_for_status()
            st.session_state.questions = response.json()
            st.session_state.current_question = 0
            st.session_state.answers = {}
            st.session_state.test_started = True
            logger.info("Test started with %d questions", len(st.session_state.questions))
        except requests.RequestException as e:
            st.error(f"Failed to fetch questions: {str(e)}")
            logger.error("Test questions fetch error: %s", str(e))

    if st.session_state.test_started and st.session_state.questions:
        question = st.session_state.questions[st.session_state.current_question]
        st.write(
            f"**Question {st.session_state.current_question + 1} of {len(st.session_state.questions)}**: {question['question']}")

        answer = st.radio("Select your answer:", question["options"], key=f"q_{st.session_state.current_question}")

        if st.button("Submit Answer"):
            if answer:
                payload = {"question_id": question["id"], "answer": answer}
                try:
                    response = requests.post(f"{BASE_URL}/check_answer", json=payload)
                    response.raise_for_status()
                    result = response.json()
                    st.session_state.answers[question["id"]] = result
                    st.success(f"Answer submitted: {result['result']}")
                    if not result["is_correct"]:
                        st.write(f"**Correct Answer**: {result['correct_answer']}")
                        st.write(f"**Explanation**: {result['explanation_ru']}")
                    logger.info("Answer submitted for question %d: %s", question["id"], result["result"])

                    st.session_state.current_question += 1
                    if st.session_state.current_question >= len(st.session_state.questions):
                        st.write("**Test Completed!**")
                        score = sum(1 for ans in st.session_state.answers.values() if ans["result"] == "Correct")
                        st.write(
                            f"**Your Score**: {score}/{len(st.session_state.questions)} ({score / len(st.session_state.questions) * 100:.1f}%)")
                        st.session_state.test_started = False
                        st.session_state.questions = []
                        logger.info("Test completed with score: %d/%d", score, len(st.session_state.answers))
                    st.rerun()
                except requests.RequestException as e:
                    st.error(f"Failed to submit answer: {str(e)}")
                    logger.error("Answer submission error: %s", str(e))
            else:
                st.warning("Please select an answer.")
                logger.warning("Empty answer submitted")


def words_tab():
    st.header("CEFR Vocabulary")
    st.write("Learn random words based on CEFR levels (A1-C2).")

    level = st.selectbox("Select CEFR level:", ["A1", "A2", "B1", "B2", "C1", "C2"])
    count = st.number_input("Number of words:", min_value=1, max_value=50, value=20)

    if st.button("Get Words"):
        payload = {"level": level, "count": count}
        try:
            response = requests.post(f"{BASE_URL}/get_words", json=payload)
            response.raise_for_status()
            words = response.json()
            st.success(f"Fetched {len(words)} words for {level} level!")
            # Display words in a table
            st.table([
                {"English": word["english"], "Russian": word["russian"]}
                for word in words
            ])
            logger.info("Fetched %d words for level %s", len(words), level)
        except requests.RequestException as e:
            st.error(f"Failed to fetch words: {str(e)}")
            logger.error("Words fetch error: %s", str(e))


def main():
    st.title("Suzy AI Assistant")
    st.write("Your personal assistant for learning English!")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Translation", "Grammar Check", "Test", "CEFR Vocabulary"])

    with tab1:
        translation_tab()
    with tab2:
        grammar_check_tab()
    with tab3:
        test_tab()
    with tab4:
        words_tab()


if __name__ == "__main__":
    main()
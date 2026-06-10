import streamlit as st
from chatbot import ask

st.set_page_config(
    page_title="HUST Student Assistant",
    page_icon="🎓"
)

st.title("🎓 HUST Student Assistant")

question = st.text_input(
    "Nhập câu hỏi về học vụ:"
)

if question:

    result = ask(question)

    st.subheader("Trả lời")

    st.write(result["answer"])

    st.subheader("Nguồn tham khảo")

    for src in result["sources"]:
        st.write(src)
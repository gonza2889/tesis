import streamlit as st
def main():
    st.set_page_config("Chat with Multiple PDFs", page_icon=":books:")
    st.header("Chat with Multiple PDFs :books:")
    with st.sidebar:
        st.header("Chat with PDF 💬")
        st.title("LLM Chatapp using LangChain")
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload the PDF Files here and Click on Process", accept_multiple_files=True)
        st.button("Process")

        st.markdown('''
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models) LLM Model
        ''')
        st.write('Do Checkout the YouTube Channel as well for amazing content [Muhammad Moin](https://www.youtube.com/channel/UC--6PuiEdiQY8nasgNluSOA)')
if __name__ == "__main__":
    main()
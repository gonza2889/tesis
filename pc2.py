import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template
embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME')
import streamlit_mermaid as stmd
# import streamlit as st

code = """
graph TD
    A --> B
"""




os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def click_evaluarRespueseta():
    st.session_state.evaluarRespuesta = True


def main():
    # os.write(1, b'-------------- Something was executed. -------------- \n')
    load_dotenv()
    st.set_page_config("Tutor Virtual", page_icon=":books:")

    st.sidebar.header("Bases de Datos 1")
    # st.sidebar.title("Seleccione un capítulo")

    st.write(css, unsafe_allow_html = True)
    st.header("Generador de Modelos Entidad-Relación :books:")

    user_answer = st.text_area("Escribe descripción del diagrama MER:")
    st.button('Crear Diagrama', on_click = click_evaluarRespueseta)

    stmd.st_mermaid(code)






if __name__ == "__main__":
    main()
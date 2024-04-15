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
from htmlTemplates import css, bot_template, user_template
embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=20, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=OpenAIEmbeddings()
    #embeddings=HuggingFaceEmbeddings(model_name = embedding_model_name)
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vectorstore.as_retriever(),memory=memory)
    return conversation_chain

def handle_user_input(user_question):
    response=st.session_state.conversation({'question':user_question})
    #st.write(response)
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2==0:
            #st.write(message)
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            #st.write(message)
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def main():
    load_dotenv()
    st.set_page_config("Tutor Virtual", page_icon=":books:")

    local_folder_path = "Teoricos"
    files = list_files(local_folder_path)
    st.sidebar.header("Files in the Local Folder")
    selected_files = st.sidebar.multiselect("Select files to upload", files)
    st.write("Selected Files:")
    for file in selected_files:
        st.write(f"- {file}")

    st.write(css, unsafe_allow_html=True)
    st.header("Tutor Virtual :books:")
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None
    user_question = st.text_input("Ask a question from your documents")
    if user_question:
        handle_user_input(user_question)
    
    # Carga de los pdfs
    # pdf_docs = st.file_uploader("Upload the PDF Files here and Click on Process", accept_multiple_files=True)
    
    
    with st.sidebar:
        st.header("Bases de Datos 1")
        st.title("Seleccione un capítulo")
        cap1 = st.button('Capítulo 1')
        cap2 = st.button('Capítulo 2')
        cap3 = st.button('Capítulo 3')


        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload the PDF Files here and Click on Process", accept_multiple_files=True)
        # st.markdown('''
        # - [Streamlit](https://streamlit.io/)
        # - [LangChain](https://python.langchain.com/)
        # - [OpenAI](https://platform.openai.com/docs/models) LLM Model
        # ''')
        st.write('Do Checkout the YouTube Channel as well for amazing content [Muhammad Moin](https://www.youtube.com/channel/UC--6PuiEdiQY8nasgNluSOA)')
        if st.button('Process'):
            with st.spinner("Processing"):
                #Extract Text from PDF
                raw_text = get_pdf_text(pdf_docs)
                #Split the Text into Chunks
                text_chunks = get_text_chunks(raw_text)
                #Create Vector Store
                vectorstore=get_vector_store(text_chunks)
                # Create Conversation Chain
                st.session_state.conversation=get_conversation_chain(vectorstore)
                st.success("Done!")
    if cap1:
        st.write('cap 1')
    if cap2:
        st.write('cap 2')
    if cap3:
        st.write('cap 3')




if __name__ == "__main__":
    main()
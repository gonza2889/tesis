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

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Carga el texto de documentos PDF
def cargar_texto_desde_pdf(ruta_pdf):
    # st.write(ruta_pdf)
    texto = ""
    with open(ruta_pdf, "rb") as archivo_pdf:
        lector_pdf = PdfReader(archivo_pdf)
        for pagina in lector_pdf.pages:
            texto += pagina.extract_text()
    return texto

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

def handle_user_input(user_question, ocultarPregunta):
    response = st.session_state.conversation({'question': user_question})
    # st.write(response)
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2==0 and ocultarPregunta == 'Pregunta':
            st.write('Pregunta:')
            # st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        elif ocultarPregunta == 'Respuesta':
            st.write('Respuesta:')
        else:
            #st.write(message)
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def handle_user_answer(user_answer, ocultarPregunta):
    response = st.session_state.conversation({'question': user_answer})
    st.session_state.chat_history = response['chat_history']
    st.write(bot_template.replace("{{MSG}}", st.session_state.chat_history[-1].content), unsafe_allow_html=True)


def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# def generar_preguntas(texto, num_preguntas=5):
#     response = openai.Completion.create(
#         engine="text-davinci-002",  # O el modelo más reciente disponible
#         prompt=f"Basado en el siguiente texto, genera {num_preguntas} preguntas interesantes para un examen: \n\n'{texto}'",
#         max_tokens=150
#     )
#     return response.choices[0].text.strip()

def generar_preguntas(texto, num_preguntas=1):
    response = st.session_state.conversation({'question': user_question})

# # Asumiendo que `texto_pdf` es el texto extraído del paso 1
# preguntas = generar_preguntas(texto_pdf)
# print(preguntas)

def carga_pdf(selected_files):
    raw_text = ""
    # CARGA DE LOS DOCUMENTOS
    for file in selected_files:
        raw_text += cargar_texto_desde_pdf('Teoricos/' + file)
    # st.write(selected_files)
    # raw_text = cargar_texto_desde_pdf('Teoricos/01BD1_DBMS.pdf')
    #Split the Text into Chunks
    text_chunks = get_text_chunks(raw_text)
    #Create Vector Store
    vectorstore = get_vector_store(text_chunks)
    # Create Conversation Chain
    st.session_state.conversation = get_conversation_chain(vectorstore)



def main():
    # os.write(1, b'-------------- Something was executed. -------------- \n')
    load_dotenv()
    st.set_page_config("Tutor Virtual", page_icon=":books:")

    st.sidebar.header("Bases de Datos 1")
    # st.sidebar.title("Seleccione un capítulo")

    # MULTISELECT CON LISTADO DE PDFs
    local_folder_path = "Teoricos"
    files = list_files(local_folder_path)
    # st.sidebar.header("Files in the Local Folder")
    selected_files = st.sidebar.multiselect("Seleccione un capítulo", files)
    # st.sidebar.write("Selected Files:")
    # for file in selected_files:
    #     st.sidebar.write(f"- {file}")
    if selected_files:
        carga_pdf(selected_files)

    st.write(css, unsafe_allow_html = True)
    st.header("Tutor Virtual :books:")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    # Carga de los pdfs
    # pdf_docs = st.file_uploader("Upload the PDF Files here and Click on Process", accept_multiple_files=True)


    if 'generarPregunta' not in st.session_state:
        st.session_state.generarPregunta = False
    def click_generarPregunta():
        st.session_state.generarPregunta = True

    if 'evaluarRespuesta' not in st.session_state:
        st.session_state.evaluarRespuesta = False
    def click_evaluarRespueseta():
        st.session_state.evaluarRespuesta = True

    # if st.button('Generar Pregunta'):
    #     with st.spinner("Procesando"):
    st.button('Generar Pregunta', on_click = click_generarPregunta)
    if st.session_state.generarPregunta:
        # st.write('prueba')
        prompt_template_name = PromptTemplate(
            input_variables = ['cuisine'],
            template = "Hazme una pregunta sobre el contenido del {cuisine}"
        )
        p = prompt_template_name.format(cuisine = "documento")
        # st.write(p)
        handle_user_input(p, 'Pregunta')

        user_answer = st.text_area("Escribe tu respuesta:")
        st.button('Evaluar respuesta', on_click = click_evaluarRespueseta)
        if st.session_state.evaluarRespuesta:
            # st.write(user_answer)
            prompt_template_name = PromptTemplate(
                input_variables = ['respuesta'],
                template = "Dime si la siguiente respuesta a tu pregunta es correcta: {respuesta}"
            )
            respuesta = prompt_template_name.format(respuesta = user_answer)
            handle_user_answer(respuesta, 'Respuesta')




if __name__ == "__main__":
    main()
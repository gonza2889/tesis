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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Carga el texto de documentos PDF
def cargarTextoDesdePdf(rutaPdf):
    texto = ""
    with open(rutaPdf, "rb") as archivo_pdf:
        lectorPdf = PdfReader(archivo_pdf)
        for pagina in lectorPdf.pages:
            texto += pagina.extract_text()
    return texto


def obtenerTrozosDeTexto(texto):
    splitter = CharacterTextSplitter(separator = "\n", chunk_size = 500, chunk_overlap = 20, length_function = len)
    trozos = splitter.split_text(texto)
    return trozos


def obtenerVectoresAlmacenados(trozosDeTexto):
    embeddings = OpenAIEmbeddings()
    vectores = FAISS.from_texts(trozosDeTexto, embedding = embeddings)
    return vectores


def obtenerCadenaDeConversacion(vectores):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    cadenaConversacion = ConversationalRetrievalChain.from_llm(llm = llm, retriever = vectores.as_retriever(), memory = memory)
    return cadenaConversacion


def manejarIngresoDelUsuario(preguntaDelUsuario, ocultarPregunta):
    respuesta = st.session_state.conversation({'question': preguntaDelUsuario})
    st.session_state.chat_history = respuesta['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0 and ocultarPregunta == 'Pregunta':
            st.write('Pregunta:')
        elif ocultarPregunta == 'Respuesta':
            st.write('Respuesta:')
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def manejarRespuestaDelUsuario(respuestaUsuario, ocultarPregunta):
    respuesta = st.session_state.conversation({'question': respuestaUsuario})
    st.session_state.chat_history = respuesta['chat_history']
    st.write(bot_template.replace("{{MSG}}", st.session_state.chat_history[-1].content), unsafe_allow_html=True)


def listaDeArchivos(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def cargaPdf(archivosSeleccionados):
    textoRaw = ""
    # Carga de los documentos
    for file in archivosSeleccionados:
        textoRaw += cargarTextoDesdePdf('Teoricos/' + file)
    # Split the Text into Chunks
    trozosDeTexto = obtenerTrozosDeTexto(textoRaw)
    # Create Vector Store
    vectores = obtenerVectoresAlmacenados(trozosDeTexto)
    # Create Conversation Chain
    st.session_state.conversation = obtenerCadenaDeConversacion(vectores)


def main():
    load_dotenv()

    st.sidebar.header("Bases de Datos 3")

    # Multiselec con listado de PDFs
    rutaCarpeta = "Teoricos"
    files = listaDeArchivos(rutaCarpeta)
    files.remove(".DS_Store")
    files.sort()
    archivosSeleccionados = st.sidebar.multiselect("Selecciona el o los capitulos sobre los que quieres recibir una pregunta:", files)

    if archivosSeleccionados:
        cargaPdf(archivosSeleccionados)

    st.write(css, unsafe_allow_html = True)

    st.header("Tutor Virtual IAGE")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if 'generarPregunta' not in st.session_state:
        st.session_state.generarPregunta = False
    def clicGenerarPregunta():
        st.session_state.generarPregunta = True

    if 'evaluarRespuesta' not in st.session_state:
        st.session_state.evaluarRespuesta = False
    def click_evaluarRespueseta():
        st.session_state.evaluarRespuesta = True

    if 'generarMultipleOpcion' not in st.session_state:
        st.session_state.generarMultipleOpcion = False
    def click_generarMultipleOpcion():
        st.session_state.generarMultipleOpcion = True


    st.button('Generar Pregunta', on_click = clicGenerarPregunta)
    if st.session_state.generarPregunta:
        promptPregunta = PromptTemplate(
            input_variables = ['doc'],
            template = "Hazme una pregunta sobre el contenido del {doc}"
        )
        prompt = promptPregunta.format(doc = "documento")
        manejarIngresoDelUsuario(prompt, 'Pregunta')

        respuestaUsuario = st.text_area("Escribe tu respuesta (si no la sabes deja el campo en blanco y presiona 'Evaluar respuesta'):")
        st.button('Evaluar respuesta', on_click = click_evaluarRespueseta)
        if st.session_state.evaluarRespuesta:
            if respuestaUsuario == '':
                promptRespuesta = PromptTemplate(
                    input_variables = [],
                    template = "Dime la respuesta correcta a tu pregunta anterior. Al final indica el nombre del documento y el número de página en la que encuentro la respuesta."
                )
            else:
                promptRespuesta = PromptTemplate(
                    input_variables = ['respuesta'],
                    template = "Dime si la siguiente respuesta a tu pregunta es correcta: {respuesta}. Si no es correcta, dime explícitamente que no es correcta, y escribe la respuesta correta. Siempre dime en qué página del documento se encuentra esa información."
                )
            respuesta = promptRespuesta.format(respuesta = respuestaUsuario)
            manejarRespuestaDelUsuario(respuesta, 'Respuesta')


if __name__ == "__main__":
    main()
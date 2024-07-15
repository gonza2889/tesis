import streamlit as st
import streamlit.components.v1 as components
import json
import requests
import streamlit as st
from langchain.llms import OpenAI


url = "https://api.openai.com/v1/chat/completions"

st.set_page_config(page_title="Mermaid", layout="wide")

st.title('Generador de Diagramas MER')
st.write('Ingresa el texto que describe la realidad que necesita ser diagramada como Modelo Entidad Relacion y luego hz clic en la flechita.')


if 'merm_code' not in st.session_state:
    st.session_state.merm_code = None
# Estado persistente para almacenar mensajes.
if "messages" not in st.session_state.keys(): # Inicializar el historial de mensajes de chat
    st.session_state.messages = [{"role": "assistant", "content": "Escribe tu descripción y haz clic en la flecha o presiona Enter."}]


def extraerSegmento(text, start_word, end_word = None):
    indiceInicio = text.find(start_word)
    
    if indiceInicio == -1:
        return None
    
    if end_word:
        indiceFinal = text.find(end_word, indiceInicio + len(start_word))
        if indiceFinal != -1 and indiceInicio < indiceFinal:
            return text[indiceInicio + len(start_word):indiceFinal].strip()
    
    else:
        return text[indiceInicio + len(start_word):].strip()
    return None


def getModelCompletion(message: str):
    payload = json.dumps({
      "messages": message,
      "model": 'gpt-3.5-turbo'
    })

    headers = {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer '
    }
    
    response = requests.request("POST", url, headers = headers, data = payload)

    return json.loads(response.text)["choices"][0]["message"]["content"]


col1, col2 = st.columns([10,10])

with col1:
    if preguntaUruario := st.chat_input(""):
        st.session_state.messages.append({"role": "user", "content": preguntaUruario})

    for message in st.session_state.messages: # Mostrar mensajes anteriores
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if preguntaUruario:
            chatPrompt = [
                {
                    "role": "system",
                    "content": "Eres un asistente de Mermaid especializado en realizar codigo Mermaid."
                },
                {
                    "role": "user",
                    "content": 'Ejemplo: \nerDiagram\n    ESTUDIANTE ||--o{ GRUPO-TRABAJO-1 : "forma"\n    GRUPO-TRABAJO-1 ||--|| GRUPO-TRABAJO-2 : "se fusiona en"\n    ESTUDIANTE {\n        int numero "Número de estudiante"\n    }\n    GRUPO-TRABAJO-1 {\n        string id "Identificador de grupo 1"\n    }\n    GRUPO-TRABAJO-2 {\n        string id "Identificador de grupo 2"\n    }'
                },
                {
                    "role": "user",
                    "content": "Ejemplo: \n erDiagram\n    Materia {\n        string nombre\n        int puntaje\n        string codigo\n    }\n    Profesor {\n        string nombre\n        string cedula_identidad\n        string telefono\n        string direccion\n    }\n    Clase {\n        string salon\n        string hora_consulta\n    }\n    Materia ||--o{ Profesor : DictadaPor\n    Profesor }o--|| Clase : Dicta"
                },
                {
                    "role": "user",
                    "content": "Ejemplo: \nerDiagram\n    Jugador {\n        int numero_inscripcion\n        int cantidad_torneos_ganados\n    }\n    Partido {\n        date fecha\n        string hora\n        string cancha\n    }\n    Dama {\n        float estatura\n    }\n    Caballero {\n        float peso\n    }\n    Pareja {\n        string nombre\n    }\n    Jugador ||--o{ Pareja : Pertenecer\n    Dama }o--|| Pareja : Integrar\n    Caballero }o--|| Pareja : Integrar\n    Partido ||--|| Pareja : Jugar"
                },
                {
                    "role": "user",
                    "content": f"Puedes darme el codigo mermaid para representar un diagrama entidad relacion para este ejercicio (necesito que me respondas solo con el código, sin ninguna palabra extra. Al final de la respuesta ponle 'END```'):\n {preguntaUruario}\n"
                } # END```
            ]
   
    # Si el último mensaje no es del asistente, generar una nueva respuesta
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = getModelCompletion(chatPrompt)
                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message) # Agregar respuesta al historial de mensajes
                st.session_state.merm_code = extraerSegmento(response, "mermaid", "END")


with col2:
    def mermaid(code: str) -> None:
        components.html(
            f"""
            <div>
                <pre class="mermaid">
                    {code}
                </pre>
                <button id="downloadSvgBtn">Download as SVG</button>
                <button id="downloadPngBtn">Download as PNG</button>
            </div>

            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true }});

                document.getElementById('downloadSvgBtn').addEventListener('click', function() {{
                    const svg = document.querySelector('.mermaid svg');
                    const svgData = new XMLSerializer().serializeToString(svg);
                    const blob = new Blob([svgData], {{ type: 'image/svg+xml' }});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'mermaid_diagram.svg';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                }});

                document.getElementById('downloadPngBtn').addEventListener('click', function() {{
                    const svg = document.querySelector('.mermaid svg');
                    const canvas = document.createElement('canvas');
                    const context = canvas.getContext('2d');
                    const svgData = new XMLSerializer().serializeToString(svg);
                    const img = new Image();
                    img.onload = function() {{
                        context.drawImage(img, 0, 0);
                        const pngData = canvas.toDataURL('image/png');
                        const downloadLink = document.createElement('a');
                        downloadLink.href = pngData;
                        downloadLink.download = 'mermaid_diagram.png';
                        downloadLink.click();
                    }};
                    img.src = 'data:image/svg+xml;base64,' + btoa(svgData);
                }});
            </script>
            """,
            height=2000
        )

    if st.session_state.merm_code:
        mermaid(
            f"""
            {st.session_state.merm_code}
            """
        )
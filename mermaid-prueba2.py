import streamlit as st
import streamlit.components.v1 as components
import json
import requests
from langchain.llms import OpenAI



st.set_page_config(page_title="Mermaid", layout="wide")

# Title of your Streamlit app
st.title('Text-to-Graph Converter')
# An explanatory comment or description about your app
st.write('This app converts your text-based processes into visual Mermaid graphs. Simply input your process in the text box below and press the button to generate the graph. You can then download the generated graph as either an SVG or a PNG file.')

if 'merm_code' not in st.session_state:
    st.session_state.merm_code = None
# Persistent state to store messages
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question!"}]

# API key
API_KEY = "jj"
model_name = "gpt-4"
url = "https://api.openai.com/v1/completions"

def extract_segment(text, start_word, end_word=None):
    start_index = text.find(start_word)
    
    if start_index == -1:
        return None
    
    if end_word:
        end_index = text.find(end_word, start_index + len(start_word))
        if end_index != -1 and start_index < end_index:
            return text[start_index + len(start_word):end_index].strip()
    
    else:
        return text[start_index + len(start_word):].strip()
    return None


def get_model_completion(message: str):
    llm = OpenAI(temperature=0.7, openai_api_key='')
    st.write(st.info(llm(message)))
    return st.info(llm(message))

#     payload = json.dumps({
#       "messages": message,
#       "temperature": 0.2
# #       "top_p": 1,
# #       "frequency_penalty": 0,
# #       "presence_penalty": 0,
# #       "max_tokens": 60,
# #       "stop": None
#     })

#     body: JSON.stringify({
#         model: "text-davinci-003"
#     })

#     headers = {
#     #   'api-key': API_KEY,
#       'Content-Type': 'application/json',
#       "Authorization": "Bearer {API_KEY}"
#     }
    
#     response = requests.request("POST", url, headers=headers, body=body, data=payload)

#     st.write(response.text)

#     return json.loads(response.text)["choices"][0]["message"]['content']

col1,col2=st.columns([10,10])

with col1:
    if user_question := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})


    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_question:
            chat_prompt = [{
                                "role": "system",
                                "content": "You are a Mermaid assistant specialized in writing Mermaid code. Always provide an answer delimited by ```mermaid -code- END```. If you need to include a command like git commit -m \"example\", please use single quotes instead of double quotes, like so: git commit -m 'example'. "
                            },
                            {
                                "role": "user",
                                "content": f"Example: '\ngraph LR\n    S1[\"Kill Minions\"] --> Q1\n    S2[\"Kill Jungle Monsters\"] --> Q2\n    S3[\"Kill Opponent Champions\"] --> Q3\n    S4[\"Destroy Enemy Structures\"] --> Q4\n    S5[\"Regular Intervals\"] --> Q5\n    S6[\"Team Objectives\"] --> Q6\n    S7[\"Penalties\"] --> D1[\"Penalties (Drain)\"]\n\n    Q1 -->|Delay| P1[\"Gold Pool\"]\n    Q1 -->|Delay| P2[\"XP Pool\"]\n    Q2 -->|Delay| P1\n    Q2 -->|Delay| P2\n    Q3 -->|Delay| P1\n    Q3 -->|Delay| P2\n    Q4 -->|Delay| P1\n    Q4 -->|Delay| P2\n    Q5 -->|Delay| P1\n    Q5 -->|Delay| P2\n    Q6 -->|Delay| P1\n    Q6 -->|Delay| P2\n\n    P1 -->|Gold| G1[\"Gate for Purchases\"]\n    P1 -->|Gold| G2[\"Gate for Game End\"]\n    P2 -->|XP| V2[\"XP Converter (Level Up)\"]\n\n    V2 -->|Convert| P4[\"Level Pool\"]\n    P4 -->|Register| R2[\"Level Register\"]\n\n    G1 -->|Purchase| V1[\"Shop (Converter)\"]\n    V1 -->|Convert| P3[\"Items Pool\"]\n    P3 -->|Register| R1[\"Items Register\"]\n    P3 -->|Drain| D1[\"Sell Items (Drain)\"]\n\n    G2 -->|Game End| E1[\"End Game\"]\n    P4 -->|Level| E1\n\n    P4 -->|Level| G3[\"Gate for Abilities\"]\n    G3 -->|Level Up| P5[\"Abilities Pool\"]\n    P5 -->|Register| R3[\"Abilities Register\"]\n    P5 -->|Drain| D2[\"Abilities Usage (Drain)\"]\n'"
                            },
                            {
                                "role": "user",
                                "content": f"Please based on the user input, write the Mermaid code for the required process. User input:\n {user_question}\n"
                            }]
   
    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, _ = get_model_completion(chat_prompt)
                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message) # Add response to message history
                st.session_state.merm_code = extract_segment(response, "mermaid", "END")

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
#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install transformers


# In[2]:


pip install PyPDF2


# In[3]:


pip install huggingface_hub


# In[4]:


pip install torch torchvision


# In[5]:


pip install ipywidgets


# In[6]:


pip install -U weaviate-client  # For beta versions: `pip install --pre -U "weaviate-client==4.*"`


# In[7]:


from huggingface_hub import notebook_login
notebook_login()


# In[8]:


import os
import weaviate
import PyPDF2
from transformers import AutoTokenizer, AutoModel


# In[ ]:


# Carga el texto de documentos PDF
def cargar_texto_desde_pdf(ruta_pdf):
    texto = ""
    with open(ruta_pdf, "rb") as archivo_pdf:
        lector_pdf = PyPDF2.PdfReader(archivo_pdf)
        for pagina in lector_pdf.pages:
            texto += pagina.extract_text()
    return texto

ruta_pdf = "PDFs/02BD1_MDD_ppt.pdf"
texto_pdf = cargar_texto_desde_pdf(ruta_pdf)

print(texto_pdf)


# In[ ]:


pip install nltk


# In[ ]:


import nltk
nltk.download('punkt')  # Descarga los datos necesarios para tokenizar oraciones

from nltk.tokenize import sent_tokenize

def split_into_sentences(text):
    return sent_tokenize(text)

# Ejemplo de uso:
input_text = "Este es un ejemplo de texto. Contiene dos oraciones."
sentences = split_into_sentences(input_text)
print(sentences)


# In[ ]:


python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path


# In[ ]:


# Carga el modelo Llama-2
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Carga el modelo Llama-2 preentrenado
# tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML")
# model = AutoModel.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML")


# In[ ]:


import torch


# In[ ]:


# Tokeniza el texto
tokenized_text = tokenizer(sentences, pad_token="[PAD]", padding=True, truncation=True, return_tensors="pt")

# Genera representaciones vectoriales
with torch.no_grad():
    outputs = model(tokenized_text)

# Las representaciones vectoriales están en outputs.last_hidden_state
representations = outputs.last_hidden_state


# In[ ]:


# Configura la clave de API de Weaviate
os.environ['WEAVIATE_API_KEY'] = "TP3jGvfknJ3ZIQGDthvpWs3QkGtJaN8C7e0l"


# In[37]:


# Consulta Weaviate para buscar documentos relevantes
def buscar_documentos(query):
    cliente_weaviate = weaviate.Client(
        url = "https://prueba1-dzfgg9kd.weaviate.network",  # Replace with your endpoint
        auth_client_secret=weaviate.auth.AuthApiKey(api_key="TP3jGvfknJ3ZIQGDthvpWs3QkGtJaN8C7e0l"),  # Replace w/ your Weaviate instance API key
    )
    resultados = cliente_weaviate.query.get(
        "DocumentosPDF",
        ['']
    ).with_near_text(
        {"concepts": [query]}
    ).with_limit(5).do()
    return resultados


# In[33]:


def almacenar_en_weaviate(representaciones_vectoriales):
    cliente_weaviate = weaviate.Client(
        url = "https://prueba1-dzfgg9kd.weaviate.network",  # Replace with your endpoint
        auth_client_secret = weaviate.auth.AuthApiKey(api_key="TP3jGvfknJ3ZIQGDthvpWs3QkGtJaN8C7e0l"),  # Replace w/ your Weaviate instance API key
    )
    clase_documentos_pdf = "DocumentosPDF"  # Crea esta clase en tu instancia de Weaviate

    # for idx, representacion in enumerate(representaciones_vectoriales):
    #     cliente_weaviate.data_object.create(
    #         class_name = clase_documentos_pdf,
    #         vector = representacion.tolist()
    #     )

    # for idx, representacion in enumerate(representaciones_vectoriales):
    # # Crea un objeto de datos con un vector
    #     data_object = {
    #         "name" : "MyVectorData",  # Nombre de la clase
    #         "myVectorProperty" : [0.1, 0.2, 0.3]  # Vector de ejemplo
    #     }
    #     # Guarda el objeto de datos en Weaviate
    #     cliente_weaviate.data.create(data_object, class_name="MyVectorData")

    for idx, representacion in enumerate(representaciones_vectoriales):
        cliente_weaviate.batch.add_data_object(
            {},
            class_name = clase_documentos_pdf,
            vector = representacion.tolist()
        )
    # cliente_weaviate.batch.add_data_object(
    #         # data_obj, 
    #         "MyVectorData", 
    #         vector=vector  # Provide openai vector
    #     )

    # for idx, representacion in enumerate(representaciones_vectoriales):
    #     cliente_weaviate.batch.create(
    #         clase_documentos_pdf,
    #         [
    #             {"id": f"documento_{idx}", "vector": representacion.tolist()}
    #         ]
    #     )


# In[ ]:


def generar_representaciones_vectoriales(texto):
    # Ejemplo ficticio:
    representaciones_vectoriales = []
    for oracion in texto.split("."):
        # Tokeniza y genera representación vectorial
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        entrada = tokenizer.encode(oracion, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            representacion = model(entrada).last_hidden_state.mean(dim=1).numpy()
            representaciones_vectoriales.append(representacion)
    print(representaciones_vectoriales)
    return representaciones_vectoriales


# In[29]:


client = weaviate.Client(
    url = "https://prueba1-dzfgg9kd.weaviate.network",  # Replace with your endpoint
    auth_client_secret=weaviate.auth.AuthApiKey(api_key="TP3jGvfknJ3ZIQGDthvpWs3QkGtJaN8C7e0l"),  # Replace w/ your Weaviate instance API key
)


# In[ ]:


def jprint(json_in):
    import json
    print(json.dumps(json_in, indent=2))

jprint(client.get_meta())


# In[26]:


jprint(client.schema.get())


# In[38]:


# Ejemplo de uso
if __name__ == "__main__":
    ruta_pdf = "PDFs/02BD1_MDD_ppt.pdf"
    texto_pdf = cargar_texto_desde_pdf(ruta_pdf)
    representaciones = generar_representaciones_vectoriales(texto_pdf)
    print(representaciones)
    almacenar_en_weaviate(representaciones)

    consulta_usuario = "buscar información relevante"
    resultados_busqueda = buscar_documentos(consulta_usuario)
    print(resultados_busqueda)


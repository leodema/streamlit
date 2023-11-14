import os

download_folder = "models"
folder = os.path.join(os.getcwd(), download_folder)
INPUT_DATA_PATH = "data/techcare/raw"
LLAMA2_MODEL = "llama-2-7b-chat.ggmlv3.q3_K_S.bin"
model_path = os.path.join(folder, LLAMA2_MODEL)

# 'models/llama-2-7b-chat.ggmlv3.q8_0.bin'
LLM_MODEL_PATH = os.path.join(download_folder, LLAMA2_MODEL)
LLM_MODEL_PATH = '/Users/leo/PycharmProjects/DochRetrival/models/llama-2-7b-chat.ggmlv3.q3_K_S.bin'
LLM_MODEL_PATH = '/Users/leo/PycharmProjects/ChatGPT_APi/llama-2-7b-chat.ggmlv3.q8_0.bin'
models_url = 'TheBloke/Llama-2-7B-Chat-GGML'
# 'models/llama-2-7b-chat.ggmlv3.q8_0.bin'

persist_directory = "./db"

DB_FAISS_PATH = 'vectorstore/db_faiss'
DB_FAISS_PATH = '/Users/leo/PycharmProjects/DochRetrival/vectorstore/db_faiss_delete'

en_qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

it_qa_template = """Utilizza le seguenti informazioni per rispondere alla domanda dell'utente.
Se non conosci la risposta, dì semplicemente che non lo sai, non cercare di inventare una risposta.
Contesto: {context}
Domanda: {question}
Restituisci solo la risposta utile di seguito e nient'altro.
Risposta utile:
"""

en_business_query_template = """Use the following pieces of information to answer the user's question.
Pretend you are an executive working at a large company, tasked to define the business strategy.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

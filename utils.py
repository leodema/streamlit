from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from config import LLAMA2_MODEL, download_folder
from huggingface_hub import hf_hub_download
from langchain.llms import CTransformers
from config import LLAMA2_MODEL, models_url, download_folder, LLM_MODEL_PATH
from config import LLAMA2_MODEL, models_url, download_folder, LLM_MODEL_PATH, persist_directory,  DB_FAISS_PATH
# Local CTransformers wrapper for Llama-2-7B-Chat
# from langchain.vectorstores import Chroma as langchain_chroma
from langchain.vectorstores import FAISS
# from chromadb import PersistentClient, Chroma
from config import it_qa_template, en_qa_template, en_business_query_template
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def download_llm():
    hf_hub_download(repo_id=models_url,
                    filename=LLAMA2_MODEL,
                    local_dir=download_folder)

def create_llm():
    llm = CTransformers(model=LLM_MODEL_PATH,  # Location of downloaded GGML model
                        model_type='llama',  # Model type Llama
                        config={'max_new_tokens': 2560,
                            'temperature': 0.01})
    return llm

# def build_llm():
#     # Local CTransformers model
#     llm = CTransformers(model=MODEL_BIN_PATH,
#                         model_type=MODEL_TYPE,
#                         config={'max_new_tokens': 256,
#                                 'temperature': 0.01}
#                         )
#
#     return llm


# Wrap prompt template in a PromptTemplate object
def set_qa_prompt():
    # TODO: Use the EN template for english questions
    # TODO: Transalte the data in english with google
    prompt = PromptTemplate(template=it_qa_template,
                            input_variables=['context', 'question'])
    return prompt

def set_business_prompt():
    prompt = PromptTemplate(template=en_business_query_template,
                            input_variables=['question'])
    return prompt


# Build RetrievalQA object
def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k':2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt})
    return dbqa


# Instantiate QA object form LangChain
# def setup_dbqa():
#     embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
#                                        model_kwargs={'device': 'cpu'})
#
#     # Now we can load the persisted vector store using Chroma
#     # vectordb = langchain_chroma(client=client,persist_directory=persist_directory, collection_name="techcare", embedding_function=embedding
#
#     llm = create_llm()
#     # qa = VectorDBQA.from_chain_type(llm=, chain_type="stuff", vectorstore=vectordb)
#     qa_prompt = set_qa_prompt()
#     dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)
#
#     return dbqa
#
#
# # Build RetrievalQA object
# def build_retrieval_qa(llm, prompt, vectordb):
#     dbqa = RetrievalQA.from_chain_type(llm=llm,
#                                        chain_type='stuff',
#                                        retriever=vectordb.as_retriever(search_kwargs={'k':2}),
#                                        return_source_documents=True,
#                                        chain_type_kwargs={'prompt': prompt})
#     return dbqa


# Instantiate QA object

def setup_dbqa():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vectordb = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = create_llm()
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa


def query(question):

    llm = create_llm()
    prompt = set_business_prompt()
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    output_text = llm_chain.run(question)

    return output_text
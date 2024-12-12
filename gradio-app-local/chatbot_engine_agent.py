import os
from typing import List

import langchain
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

langchain.verbose = True


def create_index() -> VectorStoreIndexWrapper:
    loader = DirectoryLoader("../app/data/", glob="**/*.txt")
    embeddings = HuggingFaceBgeEmbeddings(model_name="intfloat/multilingual-e5-large")
    # embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-multilingual-gemma2")
    # return VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
    )
    index_creator = VectorstoreIndexCreator(
        embedding=embeddings,
        vectorstore_cls=FAISS,
        text_splitter=text_splitter,
    )
    index = index_creator.from_loaders([loader])
    
    # Debugging: Print information about the created index
    print("Index created successfully!")
    print("Number of documents in index:", len(index.vectorstore.index_to_docstore_id))

    
    return index


def create_tools(index: VectorStoreIndexWrapper, llm) -> List[BaseTool]:
    vectorstore_info = VectorStoreInfo(
        vectorstore=index.vectorstore,
        name="test document",
        description="this story is written by takumi",
    )
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)
    return toolkit.get_tools()

def setup_model_and_tokenizer():
    local_model_path = "./local_model"
    if not os.path.exists(local_model_path):
        os.makedirs(local_model_path)
    # tokenizer = AutoTokenizer.from_pretrained("HODACHI/EZO-Common-9B-gemma-2-it", cache_dir=local_model_path)
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-jpn-it", cache_dir=local_model_path)
    tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-3.7b-instruct", cache_dir=local_model_path)
    offload_folder = os.getcwd()
    print(f"offload_folder: {offload_folder}")
    model = AutoModelForCausalLM.from_pretrained(
        # "HODACHI/EZO-Common-9B-gemma-2-it",
        # "google/gemma-2-2b-jpn-it",
        "llm-jp/llm-jp-3-3.7b-instruct",
        # "PrunaAI/HODACHI-EZO-Common-9B-gemma-2-it-bnb-4bit-smashed",
        torch_dtype="auto",
        offload_folder=offload_folder,
        offload_state_dict=True,
        cache_dir=local_model_path,
    )
    model.to("cuda")  # Move the model to CUDA
    return model, tokenizer

def setup_pipeline(model, tokenizer):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,  # トークン数を調整
        do_sample=True,      # ランダム性を有効化
        top_p=0.95,
        temperature=0.01,
        repetition_penalty=1.05,
        device_map="auto"
    )
    return HuggingFacePipeline(pipeline=pipe)

def setup_conversation_agent(llm: HuggingFacePipeline, index: VectorStoreIndexWrapper) -> AgentExecutor:

    tools = create_tools(index, llm)

    agent_chain = initialize_agent(
        tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, handle_parsing_errors=True,         max_iterations=3 
    )

    return agent_chain

def chat(
    message: str, history: ChatMessageHistory, agent: AgentExecutor
) -> str:
    memory = ConversationBufferMemory(
        chat_memory=history, memory_key="chat_history", return_messages=True
    )
    agent.memory = memory
    return agent.run(input=message, chat_history=history.messages)

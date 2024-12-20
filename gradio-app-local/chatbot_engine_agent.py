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
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline

langchain.verbose = True


def create_index() -> VectorStoreIndexWrapper:
    loader = DirectoryLoader("../app/output_markdown", glob="**/*.md")
    documents = loader.load()
    
    # Debugging: Print the number of documents loaded
    print(f"Number of documents loaded: {len(documents)}")
    
    embeddings = HuggingFaceBgeEmbeddings(model_name="intfloat/multilingual-e5-large")
    # embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-multilingual-gemma2")
    # return VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=50,

        separators="\n\n",
    )
    index_creator = VectorstoreIndexCreator(
        embedding=embeddings,
        vectorstore_cls=FAISS,
        text_splitter=text_splitter,
    )
    
    # Debugging: Check if documents are split correctly
    split_docs = text_splitter.split_documents(documents)
    print(f"Number of split documents: {len(split_docs)}")
    
    index = index_creator.from_documents(split_docs)
    
    # Debugging: Print information about the created index
    print("Index created successfully!")
    print("Number of documents in index:", len(index.vectorstore.index_to_docstore_id))

    return index


def create_tools(index: VectorStoreIndexWrapper, llm) -> List[BaseTool]:
    vectorstore_info = VectorStoreInfo(
        vectorstore=index.vectorstore,
        name="perl-oct-project-chat-log",
        description="真珠ＯＣＴ向けのプロジェクトのチャットログ",
    )
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)
    return toolkit.get_tools()

def setup_model_and_tokenizer():
    local_model_path = "./local_model"
    if not os.path.exists(local_model_path):
        os.makedirs(local_model_path)
    # tokenizer = AutoTokenizer.from_pretrained("HODACHI/EZO-Common-9B-gemma-2-it", cache_dir=local_model_path)
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-jpn-it", cache_dir=local_model_path)
    tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-13b-instruct", cache_dir=local_model_path)
    offload_folder = os.getcwd()
    print(f"offload_folder: {offload_folder}")
        # BitsAndBytesConfigを使用してモデルを量子化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        # "HODACHI/EZO-Common-9B-gemma-2-it",
        # "google/gemma-2-2b-jpn-it",
        # "llm-jp/llm-jp-3-3.7b-instruct",
        "llm-jp/llm-jp-3-13b-instruct",
        # "PrunaAI/HODACHI-EZO-Common-9B-gemma-2-it-bnb-4bit-smashed",
        torch_dtype="auto",    
        offload_folder=offload_folder,
        offload_state_dict=True,
        cache_dir=local_model_path,
        quantization_config=quantization_config
    )
    model.to("cuda")  # Move the model to CUDA
    return model, tokenizer

def setup_pipeline(model, tokenizer):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,  # トークン数を調整
        do_sample=True,      # ランダム性を有効化
        top_p=0.95,
        temperature=0.1,
        repetition_penalty=1.05,
        device_map="auto"
    )
    hugginfacePipe = HuggingFacePipeline(pipeline=pipe)
    return ChatHuggingFace(llm=hugginfacePipe)

def setup_conversation_agent(llm, index: VectorStoreIndexWrapper) -> AgentExecutor:
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    tools = create_tools(index, llm)

    agent_chain = initialize_agent(
        tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True, max_iterations=3 
    )

    return agent_chain

def chat(
    message: str, history: ChatMessageHistory, agent: AgentExecutor
) -> str:
    memory = ConversationBufferMemory(
        chat_memory=history, memory_key="chat_history", return_messages=True
    )
    agent.memory = memory
    # return agent.run(input=message, chat_history=history.messages)

    output = agent.run(input=message, chat_history=history.messages)
    ai_response_content = output.split("応答:")[-1].strip()

    return ai_response_content


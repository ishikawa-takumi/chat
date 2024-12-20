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
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from huggingface_hub import login

access_token = "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"
login(token=access_token)


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
        torch_dtype=torch.bfloat16,
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
        temperature=0.1,
        repetition_penalty=1.05,
        device_map="auto"
    )
    # return HuggingFacePipeline(pipeline=pipe)
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))

def main():
    model, tokenizer = setup_model_and_tokenizer()
    print("Model and tokenizer setup complete****************")
    pipeline = setup_pipeline(model, tokenizer)
    print("Model and pipeline setup complete****************")
    from langchain import hub
    prompt = hub.pull("hwchase17/react-chat-json")

    from langchain.agents import AgentExecutor, create_json_chat_agent
    from langchain.schema import HumanMessage, SystemMessage

    tools = []

    # Construct the ReAct agent
    agent = create_json_chat_agent(pipeline, tools, prompt)

    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # Create a chat history list
    chat_history = ChatMessageHistory().messages

    # Use the generate method instead of __call__
    messages = [
    SystemMessage(
        content=""
    ),
    HumanMessage(
        content="クリスマスプレゼントに何を買おうか迷っています。何かいいアイデアはありますか？"
    ),
    ]

    outputs = pipeline.invoke(messages)
    print(outputs.content)

    # Extract the message from the response
    # message = response.get("output", "No output returned")

    # print(message)

if __name__ == "__main__":
    main()

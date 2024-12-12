import os
import sys
import logging
from huggingface_hub import login
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.memory import ChatMessageHistory
from langchain.schema import HumanMessage
import torch
import langchain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader

langchain.verbose = True


# 1. ベクトルストアの準備
def setup_vectorstore() -> FAISS:
    # ドキュメント読み込み
    loader = DirectoryLoader("../app/data", glob="**/*.txt")
    documents = loader.load()

    # テキストをチャンクに分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=50,
    )
    docs = text_splitter.split_documents(documents)

    # Embeddings & VectorStore作成
    embeddings = HuggingFaceBgeEmbeddings(model_name="intfloat/multilingual-e5-large")
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore


# 2. チャット機能にRAGを追加
def chat_with_rag(pipeline: Pipeline, message: str, history: ChatMessageHistory, vectorstore: FAISS) -> str:

    # RAG用のRetrievalQAチェーンを構築
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    prompt_template = """
    以下の文書を参考に、以下の質問に対して回答してください。
    回答を出す際は、質問文を繰り返さず、答えのみを簡潔に出力してください。
    補足説明や余計な前置きは不要です。
    
    文書:
    {context}
    
    質問:
    {question}
    
    上記を参考に、以下に最終的な回答のみを出力してください。
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=pipeline,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )

    # チャット履歴を管理
    messages = history.messages
    messages.append(HumanMessage(content=message))

    # 検索と応答の生成
    result = qa_chain({"query": message})["result"]

    # 結果を出力
    return result


# Replace 'your_access_token' with your Hugging Face token
access_token = "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"
login(token=access_token)

print("Logged in successfully!")


# CUDA_LAUNCH_BLOCKINGを設定
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Add this line

# ログレベルの設定
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

def setup_model_and_tokenizer():
    local_model_path = "./local_model"
    if not os.path.exists(local_model_path):
        os.makedirs(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-jpn-it", cache_dir=local_model_path)
    offload_folder = os.getcwd()
    print(f"offload_folder: {offload_folder}")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-jpn-it",
        torch_dtype=torch.float16,
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
        max_new_tokens=256,  # トークン数を調整
        top_k=50,
        top_p=0.8,
        repetition_penalty=1.0,
        device_map="auto"
        # device="cuda"
    )
    return HuggingFacePipeline(pipeline=pipe)

def chat(pipeline: HuggingFacePipeline, message: str, chat_history: ChatMessageHistory) -> dict:
    messages = chat_history.messages
    messages.append(HumanMessage(content=message))

    ai_response = pipeline.invoke(messages)
    ai_response_content = ai_response  # Extract only the AI response content
    # ai_response_content = ai_response[0]["generated_text"].strip()

    return ai_response_content


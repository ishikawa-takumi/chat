import os
import sys
import logging
from huggingface_hub import login
from langchain_huggingface import ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Pipeline, BitsAndBytesConfig
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
    # loader = DirectoryLoader("../app/data", glob="**/*.txt")
    # loader = DirectoryLoader("../app/output_markdown", glob="**/*.md")
    loader = DirectoryLoader("../app/data", glob="**/*.txt")
    documents = loader.load()

    # テキストをチャンクに分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=16,
        chunk_overlap=8,
        # separators="\n\n",

        separators="***************************"
    )
    docs = text_splitter.split_documents(documents)

    # Embeddings & VectorStore作成
    embeddings = HuggingFaceBgeEmbeddings(model_name="intfloat/multilingual-e5-large")
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore


# 2. チャット機能にRAGを追加
def chat_with_rag(pipeline: Pipeline, message: str, history: ChatMessageHistory, vectorstore: FAISS) -> str:

    # RAG用のRetrievalQAチェーンを構築
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    # prompt_template = """
    # 以下の文書を参考に、以下の質問に対して回答してください。
    # 回答を出す際は、質問文を繰り返さず、答えのみを簡潔に出力してください。
    # 補足説明や余計な前置きは不要です。
    
    # 文書:
    # {context}
    
    # 質問:
    # {question}
    
    # 上記を参考に、以下に最終的な回答のみを出力してください。
    prompt_template = """
    以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。

    コンテキスト:
    {context}

    指示:
    {question}
    
    応答:
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
    result = qa_chain({"query": message})
    print(result)

    # get the AI response content
    ai_response_content = result["result"]

    # get content after "応答:"
    ai_response_content = ai_response_content.split("応答:")[-1].strip()

    return ai_response_content


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
    # tokenizer = AutoTokenizer.from_pretrained("HODACHI/EZO-Common-9B-gemma-2-it", cache_dir=local_model_path)
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-jpn-it", cache_dir=local_model_path)
    tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-3.7b-instruct", cache_dir=local_model_path)
    # tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-13b-instruct", cache_dir=local_model_path)
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4", cache_dir=local_model_path)
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
        "llm-jp/llm-jp-3-3.7b-instruct",
        # "llm-jp/llm-jp-3-13b-instruct",
        # "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
        # "PrunaAI/HODACHI-EZO-Common-9B-gemma-2-it-bnb-4bit-smashed",
        torch_dtype="auto",    
        offload_folder=offload_folder,
        offload_state_dict=True,
        cache_dir=local_model_path,
        # quantization_config=quantization_config
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
    # return ChatHuggingFace(llm=hugginfacePipe)
    return  hugginfacePipe

def chat(pipeline: HuggingFacePipeline, message: str, chat_history: ChatMessageHistory) -> dict:
    # messages = chat_history.messages
    # messages.append(HumanMessage(content=message))
    messages = [HumanMessage(content=message)]

    ai_response = pipeline.invoke(messages)
    ai_response_content = ai_response  # Extract only the AI response content
    # ai_response_content = ai_response[0]["generated_text"].strip()

    return ai_response_content


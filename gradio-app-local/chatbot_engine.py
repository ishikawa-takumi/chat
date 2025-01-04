import os
from typing import List
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Pipeline, BitsAndBytesConfig
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.memory import ChatMessageHistory
from langchain.schema import HumanMessage
import torch
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from janome.tokenizer import Tokenizer
import pdfplumber
from tqdm import tqdm
# langchain.verbose = True

# 日本語のトークン化
tokenizer = Tokenizer()

class JapaneseTextSplitter(CharacterTextSplitter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def split_text(self, text: str) -> List[str]:
        """日本語のテキストをトークン化して分割する"""
        tokens = [token.surface for token in tokenizer.tokenize(text)]
        # トークンを結合してチャンクを作成
        chunks = []
        current_chunk = ""
        for token in tokens:
            if len(current_chunk) + len(token) <= self._chunk_size:  # _chunk_sizeに修正
                current_chunk += token
            else:
                chunks.append(current_chunk)
                current_chunk = token
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

def extract_text_from_pdf(pdf_path):
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in tqdm(pdf.pages):
            texts.append(page.extract_text())
    return "\n".join(texts)

# 1. ベクトルストアの準備
def setup_vectorstore(directory_path: str) -> FAISS:
    parsed_data = []
    
    # # Recursively read PDF files from the directory
    # for root, _, files in os.walk(directory_path):
    #     for file in files:
    #         if file.endswith(".pdf"):
    #             pdf_file_path = os.path.join(root, file)
    #             parsed_data.extend(parse_pdf(pdf_file_path))
    pdf_path = 'data/message1.pdf'
    raw_text = extract_text_from_pdf(pdf_path)
    raw_text2 = extract_text_from_pdf('data/message2.pdf')
    raw_text = raw_text + raw_text2

    # Split text into chunks
    text_splitter = JapaneseTextSplitter(chunk_size=512, chunk_overlap=256)  # tokenizer引数を削除

    # Create embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="cl-nagoya/ruri-large")
    # vectorstore = FAISS.from_documents(docs, embeddings)
    texts = text_splitter.split_text(raw_text)
    vectorstore = FAISS.from_texts(texts, embeddings)


    return vectorstore


# 2. チャット機能にRAGを追加
def chat_with_rag(pipeline: Pipeline, message: str, history: ChatMessageHistory, vectorstore: FAISS) -> str:
    # RAG用のRetrievalQAチェーンを構築
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    qa_chain = RetrievalQA.from_chain_type(
        llm=pipeline,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
    )

    # チャット履歴を管理
    messages = history.messages
    messages.append(HumanMessage(content=message))

    # 検索と応答の生成
    result = qa_chain({"query": message})
    print(result)

    # get the AI response content
    ai_response_content = result["result"]

    return ai_response_content


# Replace 'your_access_token' with your Hugging Face token
access_token = "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"
login(token=access_token)

print("Logged in successfully!")


# CUDA_LAUNCH_BLOCKINGを設定
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Add this line

# ログレベルの設定
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

def setup_model_and_tokenizer():
    local_model_path = "./local_model"
    if not os.path.exists(local_model_path):
        os.makedirs(local_model_path)
    # tokenizer = AutoTokenizer.from_pretrained("HODACHI/EZO-Common-9B-gemma-2-it", cache_dir=local_model_path)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-jpn-it", cache_dir=local_model_path)
    # tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-3.7b-instruct", cache_dir=local_model_path)
    # tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-13b-instruct", cache_dir=local_model_path)
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4", cache_dir=local_model_path)
    # tokenizer = AutoTokenizer.from_pretrained("cyberagent/calm3-22b-chat", cache_dir=local_model_path)
    offload_folder = os.getcwd()
    print(f"offload_folder: {offload_folder}")
        # BitsAndBytesConfigを使用してモデルを量子化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        # "HODACHI/EZO-Common-9B-gemma-2-it",
        "google/gemma-2-2b-jpn-it",
        # "llm-jp/llm-jp-3-3.7b-instruct",
        # "llm-jp/llm-jp-3-13b-instruct",
        # "cyberagent/calm3-22b-chat",
        # "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
        # "PrunaAI/HODACHI-EZO-Common-9B-gemma-2-it-bnb-4bit-smashed",
        torch_dtype="auto",    
        offload_folder=offload_folder,
        offload_state_dict=True,
        cache_dir=local_model_path,
        quantization_config=quantization_config,
        # device_map="auto"
    )
    model.to("cuda:0")  # Move the model to CUDA  
    return model, tokenizer

def setup_pipeline(model, tokenizer):
    pipe = pipeline(
        "text-generation",
        # "question-answering",
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


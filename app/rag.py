import logging
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core import Prompt
from llama_index.core.postprocessor import SimilarityPostprocessor


from typing import Any, List

from huggingface_hub import login

# Replace 'your_access_token' with your Hugging Face token
access_token = "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"
login(token=access_token)

print("Logged in successfully!")


# CUDA_LAUNCH_BLOCKINGを設定
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ログレベルの設定
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

def setup_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-jpn-it")
    offload_folder = os.getcwd()
    print(f"offload_folder: {offload_folder}")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-jpn-it",
        torch_dtype=torch.float16,
        device_map="cuda:0",
        offload_folder=offload_folder,
        offload_state_dict=True
    )
    return model, tokenizer

def setup_pipeline(model, tokenizer):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,  # トークン数を調整
        temperature=0.3,     # サンプリングの多様性
        do_sample=True,      # ランダム性を有効化
        top_k=50,
        top_p=0.8,
        repetition_penalty=1.0
    )

    def generate_with_logging(text, **kwargs):
        print(f"Input to LLM: {text}")
        result = pipe(text, **kwargs)
        # 冗長な出力を整理
        # cleaned_result = result[0][0]["generated_text"].split("Answer:")[-1].strip()
        print(f"Output from LLM: {result}")
        return result

    class WrappedPipeline:
        def __init__(self, pipeline):
            self.pipeline = pipeline

        def __call__(self, text, **kwargs):
            return generate_with_logging(text, **kwargs)

        @property
        def task(self):
            return self.pipeline.task

    return HuggingFacePipeline(pipeline=WrappedPipeline(pipe))



def setup_embeddings():
    class HuggingFaceQueryEmbeddings(HuggingFaceEmbeddings):
        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return super().embed_documents(["passage: " + text for text in texts])

        def embed_query(self, text: str) -> List[float]:
            return super().embed_query("query: " + text)
    kwargs = {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }
    return LangchainEmbedding(HuggingFaceQueryEmbeddings(
        model_name="intfloat/multilingual-e5-large",
    ))

def setup_service_context(llm, embed_model, tokenizer):
    text_splitter = SentenceSplitter(
        chunk_size=256,
        paragraph_separator="\n\n",
        chunk_overlap=50,
        tokenizer=tokenizer.encode
    )
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = text_splitter
    Settings.num_output = 256
    Settings.context_window = 8192

def load_documents(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"ディレクトリが見つかりません: {directory}")
    return SimpleDirectoryReader(directory).load_data()


def create_index(documents, service_context):
    return VectorStoreIndex.from_documents(documents, service_context=service_context)

custom_prompt = Prompt(template="""
以下はコンテキスト情報です:
---------------------
{context_str}

質問:
{query_str}
---------------------
質問に基づいて、簡潔かつ正確な回答を提供してください。
コンテキスト情報や質問内容を繰り返さないでください。
""")

def main(data_dir="./data"):
    model, tokenizer = setup_model_and_tokenizer()
    llm = setup_pipeline(model, tokenizer)
    embed_model = setup_embeddings()
    service_context = setup_service_context(llm, embed_model, tokenizer)

    current_dir = os.getcwd()
    documents = load_documents(os.path.join(current_dir, data_dir))
    index = create_index(documents, service_context)

    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        logging.info("GPU is enabled and available.")
        logging.info(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("GPU is not enabled or not available.")

    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer()
    
    # assemble query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
    )

    # query_engine = index.as_query_engine(text_qa_template=custom_prompt)

    while True:
        try:
            user_input = input("質問を入力してください (終了するには 'exit' と入力): ").strip()
            if not user_input:
                print("入力が空です。再度入力してください。")
                continue
            if user_input.lower() == 'exit':
                break
            response = query_engine.query(user_input)
            print(response)
        except Exception as e:
            logging.error(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()

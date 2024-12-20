from huggingface_hub import HfApi
from huggingface_hub import login
access_token = "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"
login(token=access_token)
api = HfApi()

model_id = "umiumi815/EZO-Common-9B-gemma-2-it_Q4_K_M-gguf"
api.create_repo(model_id, exist_ok=True, repo_type="model")
api.upload_file(
    path_or_fileobj="../models/EZO-Common-9B-gemma-2-it_Q4_K_M.gguf",
    path_in_repo="EZO-Common-9B-gemma-2-it_Q4_K_M.gguf",
    repo_id=model_id,
)

print("Model uploaded successfully!")
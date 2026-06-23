from huggingface_hub import snapshot_download

repo_id = "WangariLM/distilbert-sst2-sentiment"

snapshot_download(
    repo_id=repo_id,
    local_dir="models/distilbert-sst2-final"
)

print(f"Model downloaded to models/distilbert-sst2-final")

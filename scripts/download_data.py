import os
# https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4
def download_datasets():
    # download "data/countdown/countdown_train.parquet" to data/countdown/countdown_train.parquet
    url = "https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4/resolve/main/data/train-00000-of-00001.parquet"
    os.makedirs("data/countdown", exist_ok=True)
    os.system(f"wget {url} -O data/countdown/train.parquet")

if __name__ == "__main__":
    download_datasets()
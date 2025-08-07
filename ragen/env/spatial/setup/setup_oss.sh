# this script is used to setup environment for gpt-oss
# Currently require H100 and CUDA 12.8 to execute


# update CUDA to 12.8
sudo rm -f /etc/apt/sources.list.d/archive_uri-https_developer_download_nvidia_com_compute_cuda_repos_ubuntu2204_x86_64_-jammy.list
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list
## update software index
sudo apt-get update
## install CUDA toolkit
sudo apt-get install -y cuda-toolkit-12-8
## link CUDA to 12.8
sudo ln -sfn /usr/local/cuda-12.8 /usr/local/cuda
## refresh environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# install miniconda
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_INSTALLER="/tmp/miniconda.sh"
echo "Downloading Miniconda from $MINICONDA_URL"
curl -sSL "$MINICONDA_URL" -o "$MINICONDA_INSTALLER"
bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
CONDA_PATH="$HOME/miniconda3/bin"
echo "export PATH=\"$CONDA_PATH:\$PATH\"" >> "$HOME/.bashrc"
source ~/.bashrc

# install env for oss
conda create -n oss python=3.12 -y
conda activate oss
pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128

# run vllm server, access same to openai api
vllm serve openai/gpt-oss-20b --gpu-memory-utilization 0.95


# Optional: forward port to local
ssh -L 8000:localhost:8000 -N -p PORT USER@IP_ADDRESS
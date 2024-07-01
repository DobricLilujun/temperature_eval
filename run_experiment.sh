#!/bin/bash -l
#SBATCH -J Singularity_Jupyter_parallel_cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1 # Tasks
#SBATCH --cpus-per-task=1 # Cores assigned to each tasks
#SBATCH --time=0-10:00:00
#SBATCH -p gpu
#SBATCH --qos=default
#SBATCH --gpus-per-task=1
#SBATCH --account=XXX
#SBATCH --mail-user=XXX@XXX
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/project/home/XXX/projects_lujun/resource/inputs/ICL/slurm_%j.out

export INPUT_PATH="resource/inputs/ICL"  # ICL

# export MODEL="Llama-2-7b-chat-hf"
# export MODEL="Llama-2-13b-chat-hf"
# export MODEL="Llama-2-70b-chat-hf"
# export MODEL="Meta-Llama-3-8B-Instruct"
# export MODEL="Meta-Llama-3-70B-Instruct"
# export MODEL="Mistral-7B-Instruct-v0.2"
# export MODEL="Mixtral-8x7B-Instruct-v0.1"
export MODEL="Mixtral-8x22B-Instruct-v0.1"

export JOB_ID=$SLURM_JOB_ID

export SERVER_NAME="my_server_${MODEL}"  #  server name for instance
export LOADING_QUANTIZATION="True" # if quantization
export LOAD_IN_4BITS="True"  # if 4 bits?
export LOAD_IN_8BITS="False"  # if 8 bit?
export MODEL_PATH="model/${MODEL}"  
export MY_ENV="causalLLM"
export PROJECT_HOME="XXX"
export USER_FOLDER="XXX"
export SINGULARITY_CACHEDIR="$PROJECT_HOME/$USER_FOLDER/.singularity/$SLURM_JOBID"
export XDG_CONFIG_HOME="$PROJECT_HOME/$USER_FOLDER/.cache"
export VENV="$PROJECT_HOME/$USER_FOLDER/.envs/venv_cuda_${MY_ENV}"
export JUPYTER_CONFIG_DIR="$PROJECT_HOME/$USER_FOLDER/jupyter_singularity/$MY_ENV/"
export JUPYTER_PATH="$VENV/share/jupyter":"$PROJECT_HOME/$USER_FOLDER/jupyter_singularity/$MY_ENV/jupyter_path"
export JUPYTER_DATA_DIR="$PROJECT_HOME/$USER_FOLDER/jupyter_singularity/$MY_ENV/jupyter_data"
export JUPYTER_RUNTIME_DIR="$PROJECT_HOME/$USER_FOLDER/jupyter_singularity/$MY_ENV/jupyter_runtime"
export IPYTHONDIR="$PROJECT_HOME/$USER_FOLDER/ipython_singularity/$SLURM_JOBID/$MY_ENV"
export IP_ADDRESS=$(hostname -I | awk '{print $1}')
export XDG_RUNTIME_DIR=""

module load env/release/latest
module load Singularity-CE/3.8.4

mkdir -p $JUPYTER_CONFIG_DIR
mkdir -p $IPYTHONDIR
mkdir -p $SINGULARITY_CACHEDIR


if [ ! -d "$VENV" ];then
    # For some reasons, there is an issue with venv -- using virtualenv instead
    singularity exec --bind $PROJECT_HOME:$PROJECT_HOME --nv jupyter_kernel_cuda.sif python3.9 -m virtualenv $VENV --system-site-packages
    singularity run --bind $PROJECT_HOME:$PROJECT_HOME --nv jupyter_kernel_cuda.sif $VENV "python3.9 -m pip install --upgrade pip" 
    singularity run --bind $PROJECT_HOME:$PROJECT_HOME --nv jupyter_kernel_cuda.sif $VENV "python3.9 -m pip install --upgrade pip fastapi transformers torch pandas accelerate uvicorn bitsandbytes"
    singularity run --bind $PROJECT_HOME:$PROJECT_HOME --nv jupyter_kernel_cuda.sif $VENV "python3.9 -m ipykernel install --sys-prefix --name $MY_ENV --display-name $MY_ENV"
fi

singularity run --bind $PROJECT_HOME:$PROJECT_HOME --nv jupyter_kernel_cuda.sif\
    $VENV \
    "python3.9 script/run_experiments.py \
    --model $MODEL\
    --input_path $INPUT_PATH\
    --loading_quantization $LOADING_QUANTIZATION\
    --load_in_4bits $LOAD_IN_4BITS\
    --load_in_8bits $LOAD_IN_8BITS"
wait

#!/bin/bash
#SBATCH --job-name=cmr_evaluation       # Job name
#SBATCH --output=/scratch/mahantas/cross_modal_retrieval/output_logs/cmr_evaluation%j.log        # Output file (%j = job ID)
#SBATCH --mail-user=saranga.boo@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --error=/scratch/mahantas/cross_modal_retrieval/output_logs/cmr_evaluation%j.log         # Error file
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu              # Partition/queue name (adjust if needed)
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=4            # CPU cores
#SBATCH --time=00:10:00              # Max runtime (HH:MM:SS)

# Load modules (adjust to your cluster's Python/CUDA module names)
module load python/3.10.14
module load cuda/12.8

# Activate your virtual environment
source /scratch/mahantas/cross_modal_retrieval/cmr/bin/activate

# Optional: check Python and torch versions
python -c "import torch; print('Python:', __import__('sys').version); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Run the evaluation script
python main.py mode=evaluate data.fold=0

## Finish gracefully
exit 0

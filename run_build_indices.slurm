#!/bin/bash
#SBATCH --job-name=cmr_build_indices       # Job name
#SBATCH --output=/scratch/mahantas/cross_modal_retrieval/output_logs/cmr_build_indices%j.log        # Output file (%j = job ID)
#SBATCH --mail-user=saranga.boo@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --error=/scratch/mahantas/cross_modal_retrieval/output_logs/cmr_build_indices%j.log         # Error file
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu              # Partition/queue name (adjust if needed)
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=4            # CPU cores
#SBATCH --time=00-00:30:00              # Max runtime (DD-HH:MM:SS)

# Load modules (adjust to your cluster's Python/CUDA module names)
module load python/3.10.14
module load cuda/12.8

# Activate your virtual environment
source /scratch/mahantas/cross_modal_retrieval/cmr/bin/activate

# Optional: check Python and torch versions
python -c "import torch; print('Python:', __import__('sys').version); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Run the training script
python build_indices_testsets.py \
  --config_path configs \
  --config_name training \
  --checkpoint multirun/2026-03-06/15-35-33_dinov3H+_FT_ClipLoss \
  --output precomputed_indices/dinov3H+_all.pt \
  --split test \
  --batch_size 64 \
  --num_workers 4 \
  --device cuda

# python main.py data.fold=3

## Finish gracefully
exit 0

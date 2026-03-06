#!/bin/bash
#SBATCH --job-name=cmr_wandb       # Job name
#SBATCH --output=/scratch/mahantas/cross_modal_retrieval/output_logs/cmr_wandb%j.log        # Output file (%j = job ID)
#SBATCH --mail-user=saranga.boo@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --error=/scratch/mahantas/cross_modal_retrieval/output_logs/cmr_wandb%j.log         # Error file
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=std             # Partition/queue name (adjust if needed)
#SBATCH --cpus-per-task=1            # CPU cores
#SBATCH --time=00:03:00              # Max runtime (HH:MM:SS)

# Load modules (adjust to your cluster's Python/CUDA module names)
module load python/3.10.14
module load cuda/12.8

# Activate your virtual environment
source /scratch/mahantas/cross_modal_retrieval/cmr/bin/activate

# Optional: check Python and torch versions
python -c "import torch; print('Python:', __import__('sys').version); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Run the evaluation script
python export_wandb_results.py

## Finish gracefully
exit 0

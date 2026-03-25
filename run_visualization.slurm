#!/bin/bash
#SBATCH --job-name=cmr_viz       # Job name
#SBATCH --output=/scratch/mahantas/cross_modal_retrieval/output_logs/cmr_viz%j.log        # Output file (%j = job ID)
#SBATCH --mail-user=saranga.boo@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --error=/scratch/mahantas/cross_modal_retrieval/output_logs/cmr_viz%j.log         # Error file
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu              # Partition/queue name (adjust if needed)
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=4            # CPU cores
#SBATCH --time=00-00:07:00              # Max runtime (DD-HH:MM:SS)
#SBATCH --mem=32G

# Load modules (adjust to your cluster's Python/CUDA module names)
module load python/3.10.14
module load cuda/12.8

# Activate your virtual environment
source /scratch/mahantas/cross_modal_retrieval/.venv/bin/activate

# Optional: check Python and torch versions
python -c "import torch; print('Python:', __import__('sys').version); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Run the visualization script
# python main.py -m data.fold=0,1,2,3,4
python main.py 

## Finish gracefully
exit 0

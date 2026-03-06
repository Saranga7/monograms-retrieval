from sklearn.model_selection import KFold
import os
from train import Config
import glob

# Create 5-fold splits containing a text file of filenames for training and validation
def create_splits(data_directory, output_directory, n_splits = 5):
    # Get all image files in the data directory
    image_files = glob.glob(data_directory + '/*.png')  # Adjust the extension if needed
    image_files.sort()  # Ensure consistent ordering
    print(f"Found {len(image_files)} image files for splitting.")

    # Create KFold splitter
    kf = KFold(n_splits = n_splits, 
               shuffle = True, 
               random_state = Config.seed)

    for fold, (train_index, test_index) in enumerate(kf.split(image_files)):
        fold_dir = os.path.join(output_directory, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        train_files = [image_files[i] for i in train_index]
        test_files = [image_files[i] for i in test_index]

        # Save train and test splits to text files
        with open(os.path.join(fold_dir, 'train.txt'), 'w') as train_file:
            for file in train_files:
                train_file.write(f"{file}\n")

        with open(os.path.join(fold_dir, 'test.txt'), 'w') as test_file:
            for file in test_files:
                test_file.write(f"{file}\n")

if __name__ == "__main__":
    data_directory = '/scratch/mahantas/datasets/MonogramSchema_Seal_pairs/schemas'  
    output_directory = '/scratch/mahantas/cross_modal_retrieval/splits'  
    n_splits = 5
    os.makedirs(output_directory, exist_ok=True)
    create_splits(data_directory, output_directory, n_splits)

    # check for test splits being mutually exclusive
    test_files = []
    for fold in range(n_splits):
        with open(os.path.join(output_directory, f'fold_{fold}', 'test.txt'), 'r') as test_file:
            test_files.extend(test_file.read().splitlines())
    assert len(test_files) == len(set(test_files)), "Test splits are not mutually exclusive!"
    print("All test splits are mutually exclusive.")  
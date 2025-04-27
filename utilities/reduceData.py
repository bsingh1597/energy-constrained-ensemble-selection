import os
import shutil
import random

def copy_limited_files(src_dir, dst_dir, max_files=300):
    # Make sure the destination directory exists
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # Loop over all folders in the source directory
    for folder_name in os.listdir(src_dir):
        folder_path = os.path.join(src_dir, folder_name)
        
        # Only proceed if it's a directory
        if os.path.isdir(folder_path):
            dst_folder_path = os.path.join(dst_dir, folder_name)
            os.makedirs(dst_folder_path, exist_ok=True)

            # Get the list of files in the current folder
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

            # Select up to `max_files` randomly (if there are more than 300 files)
            files_to_copy = files if len(files) <= max_files else random.sample(files, max_files)

            # Copy each file to the destination folder
            for file_name in files_to_copy:
                src_file_path = os.path.join(folder_path, file_name)
                dst_file_path = os.path.join(dst_folder_path, file_name)
                shutil.copy2(src_file_path, dst_file_path)
                
            print(f"Copied {len(files_to_copy)} files from {folder_name}")

# Define your source and destination directories
src_dir = '/home/myid/bs83243/mastersProject/ILSVRC/Data/CLS-LOC/train'  # Source directory with 1000 folders
dst_dir = '/home/myid/bs83243/mastersProject/ILSVRC/Data/CLS-LOC/reduced_train'  # Destination directory

# Call the function to copy folders and limit files
copy_limited_files(src_dir, dst_dir, max_files=200)

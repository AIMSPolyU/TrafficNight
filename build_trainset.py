import os
import shutil
from PIL import Image
import random
from collections import defaultdict

def find_files(root_folder, extensions):
    file_list = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                file_list.append(os.path.join(root, file))
    return file_list

def move_files(file_list, jpg_dest_folder, txt_dest_folder):
    for jpg_file, txt_file in file_list:
        jpg_file_name = os.path.basename(jpg_file)
        txt_file_name = os.path.basename(txt_file)

        img = Image.open(jpg_file)
        img.save(os.path.join(jpg_dest_folder, jpg_file_name), format='JPEG')

        shutil.copy(txt_file, os.path.join(txt_dest_folder, txt_file_name))

def organize_files(jpg_folder, txt_folder, output_folder, train_ratio=0.8):
    train_jpg_folder = os.path.join(output_folder, 'images/train')
    val_jpg_folder = os.path.join(output_folder, 'images/val')
    train_txt_folder = os.path.join(output_folder, 'labels/train')
    val_txt_folder = os.path.join(output_folder, 'labels/val')
    
    os.makedirs(train_jpg_folder, exist_ok=True)
    os.makedirs(val_jpg_folder, exist_ok=True)
    os.makedirs(train_txt_folder, exist_ok=True)
    os.makedirs(val_txt_folder, exist_ok=True)

    jpg_files = find_files(jpg_folder, ['.jpg'])
    txt_files = find_files(txt_folder, ['.txt'])
    
    txt_name_count = defaultdict(int)
    for txt_file in txt_files:
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        txt_name_count[base_name] += 1
    
    txt_files_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in txt_files}

    paired_files = []
    for jpg_file in jpg_files:
        base_name = os.path.splitext(os.path.basename(jpg_file))[0]
        if base_name in txt_files_dict:
            paired_files.append((jpg_file, txt_files_dict[base_name]))

    random.shuffle(paired_files)

    split_index = int(len(paired_files) * train_ratio)
    train_files = paired_files[:split_index]
    val_files = paired_files[split_index:]

    move_files(train_files, train_jpg_folder, train_txt_folder)
    move_files(val_files, val_jpg_folder, val_txt_folder)

if __name__ == "__main__":
    jpg_folder = '/usr/src/TrafficNight/download'
    txt_folder = '/usr/src/TrafficNight/download'
    output_folder = '/usr/src/datasets/trafficnight'
    organize_files(jpg_folder, txt_folder, output_folder, train_ratio=0.8)

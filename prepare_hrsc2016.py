import os
import shutil
import random
import glob

def prepare_dataset(base_dir):
    all_images_dir = os.path.join(base_dir, 'AllImages')
    annotations_dir = os.path.join(base_dir, 'Annotations')
    
    os.makedirs(all_images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # 1. Gather all images
    print("Gathering images...")
    image_exts = ('*.bmp', '*.jpg', '*.png')
    copied_img_count = 0
    for ext in image_exts:
        for img_path in glob.glob(os.path.join(base_dir, '**', ext), recursive=True):
            if img_path.startswith(all_images_dir): 
                continue # already in target
            filename = os.path.basename(img_path)
            dest = os.path.join(all_images_dir, filename)
            if not os.path.exists(dest):
                shutil.copy(img_path, dest)
                copied_img_count += 1
    print(f"Copied {copied_img_count} images.")
                
    # 2. Gather all annotations
    print("Gathering annotations...")
    copied_xml_count = 0
    for xml_path in glob.glob(os.path.join(base_dir, '**', '*.xml'), recursive=True):
        if xml_path.startswith(annotations_dir): 
            continue # already in target
        filename = os.path.basename(xml_path)
        
        # Rename 'Annotations100000362.xml' to '100000362.xml'
        if filename.startswith('Annotations'):
            new_filename = filename[len('Annotations'):]
        else:
            new_filename = filename
            
        dest = os.path.join(annotations_dir, new_filename)
        if not os.path.exists(dest):
            shutil.copy(xml_path, dest)
            copied_xml_count += 1
    print(f"Copied {copied_xml_count} xml files.")
            
    # 3. Use official splits instead of random splits
    print("Copying official splits...")
    for split_file in ['train.txt', 'val.txt', 'test.txt']:
        src = os.path.join(base_dir, 'ImageSets', split_file)
        dst = os.path.join(base_dir, split_file)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied {split_file} to root.")
    print("Done!")

if __name__ == '__main__':
    prepare_dataset('../DATA/HRSC2016')

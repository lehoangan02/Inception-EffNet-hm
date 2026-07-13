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
            
    # 3. Create splits based on intersection of images and annotations
    print("Creating splits...")
    images = [os.path.splitext(f)[0] for f in os.listdir(all_images_dir) if f.endswith(('.bmp', '.jpg', '.png'))]
    xmls = [os.path.splitext(f)[0] for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    
    # Only keep IDs that have both image and annotation
    valid_ids = list(set(images).intersection(set(xmls)))
    print(f"Found {len(images)} total images in AllImages, {len(xmls)} annotations in Annotations.")
    print(f"Total valid pairs (both image and xml exist): {len(valid_ids)}")
    
    if len(valid_ids) == 0:
        print("No valid pairs found to split! Make sure images and xmls match.")
        return
        
    valid_ids.sort()
    random.seed(42)
    random.shuffle(valid_ids)
    
    total = len(valid_ids)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    train_ids = valid_ids[:train_end]
    val_ids = valid_ids[train_end:val_end]
    test_ids = valid_ids[val_end:]
    
    def write_file(filename, ids):
        filepath = os.path.join(base_dir, filename)
        with open(filepath, 'w') as f:
            for id in ids:
                f.write(id + '\n')
        print(f'Wrote {len(ids)} ids to {filepath}')
        
    write_file('train.txt', train_ids)
    write_file('val.txt', val_ids)
    write_file('test.txt', test_ids)
    print("Done!")

if __name__ == '__main__':
    prepare_dataset('../DATA/HRSC2016_dataset/HRSC2016')

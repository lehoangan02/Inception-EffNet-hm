import os
import random
import argparse

def generate_txt(data_dir):
    image_folder = os.path.join(data_dir, 'AllImages')
    if not os.path.exists(image_folder):
        print(f"Error: Could not find {image_folder}")
        return

    # Get all file names in the images folder
    image_files = os.listdir(image_folder)

    # Remove the extension (.bmp, .jpg, etc) from each file name
    image_names = [os.path.splitext(file)[0] for file in image_files if file.endswith(('.bmp', '.jpg', '.png'))]
    
    # Shuffle the names for a random split
    random.seed(42)
    random.shuffle(image_names)
    
    # Let's do a 80% train, 10% val, 10% test split
    total = len(image_names)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    train_names = image_names[:train_end]
    val_names = image_names[train_end:val_end]
    test_names = image_names[val_end:]

    def write_file(filename, names):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'w') as f:
            for name in names:
                f.write(name + '\n')
        print(f'Wrote {len(names)} names to {filepath}')

    write_file('train.txt', train_names)
    write_file('val.txt', val_names)
    write_file('test.txt', test_names)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to your HRSC dataset directory (e.g. DATA/archive)')
    args = parser.parse_args()
    generate_txt(args.data_dir)

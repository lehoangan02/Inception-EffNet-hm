import os
import xml.etree.ElementTree as ET
import numpy as np

def test_parse_gt():
    data_dir = '../DATA/HRSC2016MS'
    imagesetfile = os.path.join(data_dir, 'test.txt')
    if not os.path.exists(imagesetfile):
        print(f"File not found: {imagesetfile}")
        return
        
    with open(imagesetfile, 'r') as f:
        imagenames = [x.strip() for x in f.readlines()]
        
    if not imagenames:
        print("No images in test.txt")
        return
        
    print(f"Total images in test.txt: {len(imagenames)}")
    
    label_path = os.path.join(data_dir, 'Annotations')
    npos = 0
    for imagename in imagenames:
        xml_file = os.path.join(label_path, f'{imagename}.xml')
        if not os.path.exists(xml_file):
            print(f"Missing XML: {xml_file}")
            continue
            
        target = ET.parse(xml_file).getroot()
        objects = []
        for obj in target.iter('HRSC_Object'):
            diff_node = obj.find('difficult')
            difficult = int(diff_node.text) if diff_node is not None else 0
            objects.append({'name': 'ship', 'difficult': difficult})
            
        R = [obj for obj in objects if obj['name'] == 'ship']
        difficult_arr = np.array([x['difficult'] for x in R]).astype(bool)
        npos += sum(~difficult_arr)
        
    print(f"Total objects found: {npos}")

if __name__ == '__main__':
    test_parse_gt()

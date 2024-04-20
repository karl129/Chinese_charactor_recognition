import os
import json
import numpy as np
import struct
from PIL import Image


data_dir = './data'
train_data_dir = os.path.join(data_dir, 'HWDB1.1trn_gnt')
test_data_dir = os.path.join(data_dir, 'HWDB1.1tst_gnt')


def read_from_gnt_dir(gnt_dir=train_data_dir):
    def one_file(f):
        header_size = 10
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size:
                break
            sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
            tagcode = header[5] + (header[4] << 8)
            width = header[6] + (header[7] << 8)
            height = header[8] + (header[9] << 8)
            if header_size + width * height != sample_size:
                break
            image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
            yield image, tagcode

    for file_name in os.listdir(gnt_dir):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            with open(file_path, 'rb') as f:
                yield from one_file(f)


def create_char_dict(gnt_dir=train_data_dir):
    char_set = set()
    for _, tagcode in read_from_gnt_dir(gnt_dir):
        tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
        char_set.add(tagcode_unicode)

    char_list = list(char_set)
    char_dict = dict(zip(sorted(char_list), range(len(char_list))))
    return char_dict


def save_char_dict(char_dict):
    with open('char_dict.json', 'w') as f:
        json.dump(char_dict, f, ensure_ascii=False)


def save_images(gnt_dir, output_dir):
    counter = 0
    for image, tagcode in read_from_gnt_dir(gnt_dir):
        tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
        im = Image.fromarray(image)
        dir_name = os.path.join(output_dir, '%0.5d' % char_dict[tagcode_unicode])
        os.makedirs(dir_name, exist_ok=True)
        im.convert('RGB').save(os.path.join(dir_name, str(counter) + '.png'))
        counter += 1

def classes_txt(root, out_path, num_class=None):
    dirs = os.listdir(root)
    if not num_class:
        num_class = len(dirs)

    if not os.path.exists(out_path):
        f = open(out_path, 'w')
        f.close()

    with open(out_path, 'r+') as f:
        try:
            end = int(f.readlines()[-1].split('/')[-2]) + 1
        except:
            end = 0
        if end < num_class - 1:
            dirs.sort()
            dirs = dirs[end:num_class]
            for dir in dirs:
                files = os.listdir(os.path.join(root, dir))
                for file in files:
                    f.write(os.path.join(root, dir, file) + '\n')



if __name__ == '__main__':
    char_dict = create_char_dict()
    save_char_dict(char_dict)

    train_output_dir = os.path.join(data_dir, 'train')
    test_output_dir = os.path.join(data_dir, 'test')

    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    save_images(train_data_dir, train_output_dir)
    save_images(test_data_dir, test_output_dir)

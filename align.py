import cv2, os
import numpy as np
import argparse
from mmRegressor.preprocess_img import Preprocess
from mmRegressor.load_data import BFM
from PIL import Image

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--meta_path', type=str, required=True, help='5 landmarks file')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--name', type=str, required=True, help='dataset name')
    args = parser.parse_args()

    # read face model
    face_model = BFM('mmRegressor/BFM/BFM_model_80.mat', -1)
    lm3D = face_model.load_lm3d("mmRegressor/BFM/similarity_Lm3D_all.mat")

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    fail_f = open(args.name + '_fail.txt', 'wt')
    with open(args.meta_path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if i%50==0:
            print('\r%.3f%%...' % (((i+1)/len(lines))*100), end='')

        splits = line.strip().split()
        fname = os.path.join(args.image_path, splits[0])
        if not os.path.exists(fname):
            fail_f.write(fname+'\n')
            continue
        
        lmk = list(map(float, splits[1:]))
        lmk = np.reshape(np.array(lmk), (5,2))
        img = Image.open(fname)
        try:
            img, _ = Preprocess(img, lmk, lm3D, 224)
        except ValueError:
            fail_f.write(fname+'\n')
            continue
        
        cv2.imwrite(os.path.join(args.save_path,splits[0]), img[0])
    
    fail_f.close()
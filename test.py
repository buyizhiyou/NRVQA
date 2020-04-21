import argparse
import os
import sys
import pdb

import cv2
from joblib import load

from brisque import brisque
from niqe import niqe
from piqe import piqe

parser = argparse.ArgumentParser('Test an image')
parser.add_argument(
    '--mode', choices=['brisque', 'niqe', 'piqe'], help='iqa algorithoms,brisque or niqe or piqe')
parser.add_argument('--path', required=True, help='image path')
args = parser.parse_args()

if __name__ == "__main__":
    '''
    test conventional blindly image quality assessment methods(brisque/niqe/piqe)
    '''
    mode = args.mode
    path = args.path
    im = cv2.imread(path)
    if im is None:
        print("please input correct image path!")
        sys.exit(0)
    if mode == "piqe":
        score, _, _, _ = piqe(im)
    elif mode == "niqe":
        score = niqe(im)
    elif mode == "brisque":
        feature = brisque(im)
        feature = feature.reshape(1, -1)
        clf = load('svr_brisque.joblib')
        score = clf.predict(feature)[0]
    print("{}-----{} score:{}".format(path, mode, score))

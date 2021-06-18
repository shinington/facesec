# -*- coding: utf-8 -*-

"""
robust_evalution.py
~~~~~~~

Main entry point for the robust assessment phase of FaseSec.

"""
import argparse
import os

from face_recognition import FaceRecognition
from utils import str2bool

def main():
    args = parse_args()
    
    if args.target == None or (args.target != None and not os.path.exists(args.output)):
        # White-box attacks
        # 1. Load source face recognition model.
        source = FaceRecognition(args.mode,
            args.source,
            args.input,
            args.gallery,
            args.output,
            args.universal)
        print("##############################################################################")
        print("Load source model: {}...".format(args.source))
        source.load_model()

        # 2. Evaluate on non-adversarial data.
        print("Evaluate source model on clean data...")
        source.eval_folder()

        # 3. Evaluate by white-box attacks.
        print("Evaluate source model using {} attacks...".format(args.attack))
        source.eval_robustness(args.attack)
    if args.target != None:
        # Black-box attacks
        # 4. Load target face recognition model.
        target = FaceRecognition(args.mode,
            args.target,
            args.input,
            args.gallery,
            None,
            None)
        print("##############################################################################")
        print("Load target model: {}...".format(args.target))
        target.load_model()

        # 5. Evaluate target on non-adversarial data
        print("Evaluate target model on clean data...")
        target.eval_folder()

        # 6. Evaluate target with adversarial examples transferred from source
        print("Evaluate target model on adversarial examples transferred from source model...")
        target.eval_folder(args.output)        

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-s', '--source', 
                   help='Source model name to be evaluated.')
    p.add_argument('-t', '--target', default=None,
                   help='Target model name to be evaluated (used in black-box attacks).')
    p.add_argument('-m', '--mode',
                   help='Mode of face recognition. Can be one of [closed, open].')
    p.add_argument('-a', '--attack', 
                   help='Attack mothod Can be one of [pgd, sticker, eyeglass, facemask].')
    p.add_argument('-u', '--universal', type=str2bool, nargs='?', const=True, default=False,
                   help='Whether to produce universal adversarial examples.')
    p.add_argument('-i', '--input',
                   help='Directory of input test images.')
    p.add_argument('-o', '--output', 
                   help='Directory of output adversarial examples.')
    p.add_argument('-g', '--gallery', default=None,
                   help='Directory of gallery images (ONLY and MUST for open-set).')

    args = p.parse_args()

    if not any([args.source, args.mode, args.attack, args.input, args.output]):
        print('[!] Either a source model must be supplied [-s], '
              'or a mode [-m],'
              'or an attack must be invoked [-a],'
              'or an input directory [-i],'
              'or an output directory [-o].')
        exit()
    if args.mode == 'open' and args.gallery == None:
        print('This is open-set face recognition, please input the gallery path [-g]')
        exit()
    return args

if __name__ == '__main__':
    main()

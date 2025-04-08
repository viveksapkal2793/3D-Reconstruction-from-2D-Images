import sys
import os

# Try to ensure OpenCV is in the path 
site_packages = os.path.join(os.path.dirname(os.__file__), 'site-packages')
if site_packages not in sys.path:
    sys.path.append(site_packages)

# Check for OpenCV and print debugging info
try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"Error importing OpenCV: {e}")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    # Try a last-resort pip install
    try:
        print("Attempting to install OpenCV...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-contrib-python-headless==4.8.0.76"])
        import cv2
        print(f"Successfully installed OpenCV: {cv2.__version__}")
    except Exception as install_error:
        print(f"Failed to install OpenCV: {install_error}")
        sys.exit(1)

import numpy as np 
import pickle 
import argparse
from time import time

from utils import * 

def FeatMatch(opts, data_files=[]): 
    
    if len(data_files) == 0: 
        img_names = sorted(os.listdir(opts.data_dir))
        img_paths = [os.path.join(opts.data_dir, x) for x in img_names if \
                    x.split('.')[-1] in opts.ext]
    
    else: 
        img_paths = data_files
        img_names = sorted([x.split('/')[-1] for x in data_files])
        
    # feat_out_dir = os.path.join(opts.out_dir,'features',opts.features)
    # matches_out_dir = os.path.join(opts.out_dir,'matches',opts.matcher)

    # Use custom output directories if provided, otherwise use defaults
    if hasattr(opts, 'feat_out_dir') and opts.feat_out_dir:
        feat_out_dir = opts.feat_out_dir
    else:
        feat_out_dir = os.path.join(opts.out_dir, 'features', opts.features)
        
    if hasattr(opts, 'matches_out_dir') and opts.matches_out_dir:
        matches_out_dir = opts.matches_out_dir
    else:
        matches_out_dir = os.path.join(opts.out_dir, 'matches', opts.matcher)

    if not os.path.exists(feat_out_dir): 
        os.makedirs(feat_out_dir)
    if not os.path.exists(matches_out_dir): 
        os.makedirs(matches_out_dir)
    
    data = []
    t1 = time()
    for i, img_path in enumerate(img_paths): 
        img = cv2.imread(img_path)
        img_name = img_names[i].split('.')[0]
        img = img[:,:,::-1]

        # # Modified feature detection code with fallback
        # try:
        #     if opts.features in ['SURF', 'SIFT']:
        #         feat = getattr(cv2.xfeatures2d, '{}_create'.format(opts.features))()
        #     else:
        #         # For free algorithms like ORB, BRISK, etc.
        #         feat = getattr(cv2, '{}_{}'.format(opts.features, 'create'))()
        #     kp, desc = feat.detectAndCompute(img,None)
        # except (AttributeError, cv2.error):
        #     print(f"Warning: {opts.features} not available, falling back to ORB")
        #     feat = cv2.ORB_create()
        #     kp, desc = feat.detectAndCompute(img,None)

        try:
            if opts.features in ['SURF', 'SIFT']:
                # Try multiple ways to initialize the feature detector
                try:
                    # First try xfeatures2d
                    feat = getattr(cv2.xfeatures2d, '{}_create'.format(opts.features))()
                except (AttributeError, cv2.error):
                    # Then try direct creation (OpenCV 4.x)
                    feat = getattr(cv2, opts.features + '_create')()
            else:
                # For algorithms like ORB, BRISK, etc. with increased features for ORB
                if opts.features == 'ORB':
                    feat = cv2.ORB_create(nfeatures=10000)  # Increase from default 500
                else:
                    feat = getattr(cv2, '{}_{}'.format(opts.features, 'create'))()
            kp, desc = feat.detectAndCompute(img, None)
        except Exception as e:
            print(f"Error with {opts.features}, falling back to ORB: {str(e)}")
            feat = cv2.ORB_create(nfeatures=10000)
            kp, desc = feat.detectAndCompute(img, None)

        data.append((img_name, kp, desc))

        kp_ = SerializeKeypoints(kp)
        
        with open(os.path.join(feat_out_dir, 'kp_{}.pkl'.format(img_name)),'wb') as out:
            pickle.dump(kp_, out)

        with open(os.path.join(feat_out_dir, 'desc_{}.pkl'.format(img_name)),'wb') as out:
            pickle.dump(desc, out)

        if opts.save_results: 
            raise NotImplementedError

        t2 = time()

        if (i % opts.print_every) == 0:    
            print('FEATURES DONE: {0}/{1} [time={2:.2f}s]'.format(i+1, len(img_paths), t2-t1))

        t1 = time()

    num_done = 0 
    num_matches = ((len(img_paths)-1) * (len(img_paths))) / 2

    t1 = time()
    for i in range(len(data)): 
        for j in range(i+1, len(data)): 
            img_name1, kp1, desc1 = data[i]
            img_name2, kp2, desc2 = data[j]

            matcher = getattr(cv2,opts.matcher)(crossCheck=opts.cross_check)
            matches = matcher.match(desc1,desc2)

            matches = sorted(matches, key = lambda x:x.distance)
            matches_ = SerializeMatches(matches)

            pickle_path = os.path.join(matches_out_dir, 'match_{}_{}.pkl'.format(img_name1,
                                                                                 img_name2))
            with open(pickle_path,'wb') as out:
                pickle.dump(matches_, out)

            num_done += 1 
            t2 = time()

            if (num_done % opts.print_every) == 0: 
                print('MATCHES DONE: {0}/{1} [time={2:.2f}s]'.format(num_done, num_matches, t2-t1))

            t1 = time()
            


def SetArguments(parser): 

    #directories stuff
    parser.add_argument('--data_files',action='store',type=str,default='',dest='data_files') 
    parser.add_argument('--data_dir',action='store',type=str,default='../data/fountain-P11/images/',
                        dest='data_dir',help='directory containing images (default: ../data/\
                        fountain-P11/images/)') 
    parser.add_argument('--ext',action='store',type=str,default='jpg,png',dest='ext',
                        help='comma seperated string of allowed image extensions \
                        (default: jpg,png)') 
    parser.add_argument('--out_dir',action='store',type=str,default='../data/fountain-P11/',
                        dest='out_dir',help='root directory to store results in \
                        (default: ../data/fountain-P11)') 

    #feature matching args
    parser.add_argument('--features',action='store', type=str, default='ORB', dest='features',
                        help='[SIFT|SURF|ORB] Feature algorithm to use (default: ORB)') 
    parser.add_argument('--matcher',action='store',type=str,default='BFMatcher',dest='matcher',
                        help='[BFMatcher|FlannBasedMatcher] Matching algorithm to use \
                        (default: BFMatcher)') 
    parser.add_argument('--cross_check',action='store',type=bool,default=True,dest='cross_check',
                        help='[True|False] Whether to cross check feature matching or not \
                        (default: True)') 
    
    #misc
    parser.add_argument('--print_every',action='store', type=int, default=1, dest='print_every',
                        help='[1,+inf] print progress every print_every seconds, -1 to disable \
                        (default: 1)')
    parser.add_argument('--save_results',action='store', type=str, default=False, 
                        dest='save_results',help='[True|False] whether to save images with\
                        keypoints drawn on them (default: False)')  
    
        # Add new parameters for custom output directories
    parser.add_argument('--feat_out_dir', action='store', type=str, default='', 
                        dest='feat_out_dir', help='custom directory to store feature results')
    parser.add_argument('--matches_out_dir', action='store', type=str, default='',
                        dest='matches_out_dir', help='custom directory to store matches results')

def PostprocessArgs(opts): 
    opts.ext = [x for x in opts.ext.split(',')]
    
    opts.data_files_ = []
    if opts.data_files != '': 
        opts.data_files_ = opts.data_files.split(',')
    opts.data_files = opts.data_files_

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    SetArguments(parser)
    opts = parser.parse_args()
    PostprocessArgs(opts)

    FeatMatch(opts, opts.data_files)
import numpy as np 
import cv2 
import argparse
import pickle
import os 
from time import time
import matplotlib.pyplot as plt

from utils import * 
import pdb 

class Camera(object): 
    def __init__(self, R, t, ref): 
        self.R = R 
        self.t = t 
        self.ref = ref

class Match(object): 
    def __init__(self, matches, img1pts, img2pts, img1idx, img2idx, mask): 
        self.matches = matches
        self.img1pts, self.img2pts = img1pts, img2pts 
        self.img1idx, self.img2idx = img1idx, img2idx
        self.mask = mask

class SFM(object): 
    def __init__(self, opts): 
        self.opts = opts
        self.point_cloud = np.zeros((0,3))

        # setting up directory stuff..
        self.images_dir = os.path.join(opts.data_dir, opts.dataset, 'images')
        
        # Use custom feature directory if provided
        if hasattr(opts, 'feat_in_dir') and opts.feat_in_dir:
            self.feat_dir = opts.feat_in_dir
        else:
            self.feat_dir = os.path.join(opts.data_dir, opts.dataset, 'features', opts.features)
        
        # Use custom matches directory if provided
        if hasattr(opts, 'matches_in_dir') and opts.matches_in_dir:
            self.matches_dir = opts.matches_in_dir
        else:
            self.matches_dir = os.path.join(opts.data_dir, opts.dataset, 'matches', opts.matcher)

        # Output directories
        if hasattr(opts, 'out_cloud_dir') and opts.out_cloud_dir:
            self.out_cloud_dir = opts.out_cloud_dir
        else:
            self.out_cloud_dir = os.path.join(opts.out_dir, opts.dataset, 'point-clouds')
            
        if hasattr(opts, 'out_err_dir') and opts.out_err_dir:
            self.out_err_dir = opts.out_err_dir
        else:
            self.out_err_dir = os.path.join(opts.out_dir, opts.dataset, 'errors')

        # Create output directories if they don't exist
        os.makedirs(self.out_cloud_dir, exist_ok=True)
        if not opts.skip_reprojection:
            os.makedirs(self.out_err_dir, exist_ok=True)

        self.image_names = [x.split('.')[0] for x in sorted(os.listdir(self.images_dir)) \
                            if x.split('.')[-1] in opts.ext]

        # setting up shared parameters for the pipeline
        self.image_data, self.matches_data, errors = {}, {}, {}
        self.matcher = getattr(cv2, opts.matcher)(crossCheck=opts.cross_check)

        # Handle intrinsic matrix
        if opts.custom_intrinsics:
            # Try to load intrinsics from calibration file
            try:
                calib_path = os.path.join(opts.data_dir, opts.dataset, 'calibration', 'intrinsics.npy')
                self.K = np.load(calib_path)
                print(f"Loaded custom intrinsics matrix: \n{self.K}")
            except Exception as e:
                print(f"Error loading custom intrinsics: {e}")
                # Fallback to default
                self.K = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1]])
                print(f"Using fallback intrinsics: \n{self.K}")
        elif opts.calibration_mat == 'benchmark': 
            self.K = np.array([[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]])
        elif opts.calibration_mat == 'lg_g3': 
            self.K = np.array([[3.97*320, 0, 320], [0, 3.97*320, 240], [0, 0, 1]])
        else: 
            raise NotImplementedError
        
    def _LoadFeatures(self, name): 
        # with open(os.path.join(self.feat_dir,'kp_{}.pkl'.format(name)),'r') as f: 
        #     kp = pickle.load(f)
        # kp = DeserializeKeypoints(kp)

        # with open(os.path.join(self.feat_dir,'desc_{}.pkl'.format(name)),'r') as f: 
        #     desc = pickle.load(f)

        # return kp, desc 

        try:
            with open(os.path.join(self.feat_dir,'kp_{}.pkl'.format(name)),'rb') as f:  # Changed 'r' to 'rb' for binary reading
                kp = pickle.load(f)
            kp = DeserializeKeypoints(kp)

            with open(os.path.join(self.feat_dir,'desc_{}.pkl'.format(name)),'rb') as f:  # Changed 'r' to 'rb'
                desc = pickle.load(f)

            return kp, desc
        except (IOError, FileNotFoundError) as e:
            print(f"Warning: Failed to load features for {name}. Error: {e}")
            # Could add code here to generate features on the fly if needed
            raise

    def _LoadMatches(self, name1, name2): 
        # with open(os.path.join(self.matches_dir,'match_{}_{}.pkl'.format(name1,name2))
        #             ,'r') as f: 
        
        # Changed 'r' to 'rb' for binary reading of pickle files
        with open(os.path.join(self.matches_dir,'match_{}_{}.pkl'.format(name1,name2))
                    ,'rb') as f: 
            matches = pickle.load(f)
        matches = DeserializeMatches(matches)
        return matches

    def _GetAlignedMatches(self,kp1,desc1,kp2,desc2,matches):

        # Ensure matches indices are within the keypoint arrays bounds
        valid_matches = []
        for m in matches:
            if m.queryIdx < len(kp1) and m.trainIdx < len(kp2):
                valid_matches.append(m)
            else:
                print(f"Skipping match with invalid indices: queryIdx={m.queryIdx}, trainIdx={m.trainIdx}, kp1_len={len(kp1)}, kp2_len={len(kp2)}")
        
        if len(valid_matches) < len(matches):
            print(f"Filtered out {len(matches) - len(valid_matches)} invalid matches out of {len(matches)} total")
        
        if not valid_matches:
            raise ValueError("No valid matches found between images")
        
        img1idx = np.array([m.queryIdx for m in matches])
        img2idx = np.array([m.trainIdx for m in matches])

        #filtering out the keypoints that were matched. 
        kp1_ = (np.array(kp1))[img1idx]
        kp2_ = (np.array(kp2))[img2idx]

        #retreiving the image coordinates of matched keypoints
        img1pts = np.array([kp.pt for kp in kp1_])
        img2pts = np.array([kp.pt for kp in kp2_])

        return img1pts, img2pts, img1idx, img2idx

    def _BaselinePoseEstimation(self, name1, name2):

        kp1, desc1 = self._LoadFeatures(name1)
        kp2, desc2 = self._LoadFeatures(name2)  

        matches = self._LoadMatches(name1, name2)
        matches = sorted(matches, key = lambda x:x.distance)

        img1pts, img2pts, img1idx, img2idx = self._GetAlignedMatches(kp1,desc1,kp2,
                                                                    desc2,matches)
        
        # F,mask = cv2.findFundamentalMat(img1pts,img2pts,method=opts.fund_method,
        #                                 param1=opts.outlier_thres,param2=opts.fund_prob)

        F,mask = cv2.findFundamentalMat(img1pts,img2pts,method=opts.fund_method,
                                    ransacReprojThreshold=opts.outlier_thres,confidence=opts.fund_prob)
        
        mask = mask.astype(bool).flatten()

        E = self.K.T.dot(F.dot(self.K))
        _,R,t,_ = cv2.recoverPose(E,img1pts[mask],img2pts[mask],self.K)

        self.image_data[name1] = [np.eye(3,3), np.zeros((3,1)), np.ones((len(kp1),))*-1]
        self.image_data[name2] = [R,t,np.ones((len(kp2),))*-1]

        self.matches_data[(name1,name2)] = [matches, img1pts[mask], img2pts[mask], 
                                            img1idx[mask],img2idx[mask]]

        return R,t

    def _TriangulateTwoViews(self, name1, name2): 

        def __TriangulateTwoViews(img1pts, img2pts, R1, t1, R2, t2): 
            img1ptsHom = cv2.convertPointsToHomogeneous(img1pts)[:,0,:]
            img2ptsHom = cv2.convertPointsToHomogeneous(img2pts)[:,0,:]

            img1ptsNorm = (np.linalg.inv(self.K).dot(img1ptsHom.T)).T
            img2ptsNorm = (np.linalg.inv(self.K).dot(img2ptsHom.T)).T

            img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:,0,:]
            img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:,0,:]

            pts4d = cv2.triangulatePoints(np.hstack((R1,t1)),np.hstack((R2,t2)),
                                            img1ptsNorm.T,img2ptsNorm.T)
            pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:,0,:]

            return pts3d

        def _Update3DReference(ref1, ref2, img1idx, img2idx, upp_limit, low_limit=0): 

            ref1[img1idx] = np.arange(upp_limit) + low_limit
            ref2[img2idx] = np.arange(upp_limit) + low_limit

            return ref1, ref2

        R1, t1, ref1 = self.image_data[name1]
        R2, t2, ref2 = self.image_data[name2]

        _, img1pts, img2pts, img1idx, img2idx = self.matches_data[(name1,name2)]
        
        new_point_cloud = __TriangulateTwoViews(img1pts, img2pts, R1, t1, R2, t2)
        self.point_cloud = np.concatenate((self.point_cloud, new_point_cloud), axis=0)

        ref1, ref2 = _Update3DReference(ref1, ref2, img1idx, img2idx,new_point_cloud.shape[0],
                                        self.point_cloud.shape[0]-new_point_cloud.shape[0])
        self.image_data[name1][-1] = ref1 
        self.image_data[name2][-1] = ref2 

    def _TriangulateNewView(self, name): 
        
        for prev_name in self.image_data.keys(): 
            if prev_name != name: 
                kp1, desc1 = self._LoadFeatures(prev_name)
                kp2, desc2 = self._LoadFeatures(name)  

                prev_name_ref = self.image_data[prev_name][-1]
                matches = self._LoadMatches(prev_name,name)
                matches = [match for match in matches if prev_name_ref[match.queryIdx] < 0]

                if len(matches) > 0: 
                    matches = sorted(matches, key = lambda x:x.distance)

                    img1pts, img2pts, img1idx, img2idx = self._GetAlignedMatches(kp1,desc1,kp2,
                                                                                desc2,matches)
                    
                    # F,mask = cv2.findFundamentalMat(img1pts,img2pts,method=opts.fund_method,
                    #                                 param1=opts.outlier_thres,
                    #                                 param2=opts.fund_prob)

                    F,mask = cv2.findFundamentalMat(img1pts,img2pts,method=opts.fund_method,
                                                    ransacReprojThreshold=opts.outlier_thres,
                                                    confidence=opts.fund_prob)
                    
                    mask = mask.astype(bool).flatten()

                    self.matches_data[(prev_name,name)] = [matches, img1pts[mask], img2pts[mask], 
                                                img1idx[mask],img2idx[mask]]
                    self._TriangulateTwoViews(prev_name, name)

                else: 
                    print('skipping {} and {}'.format(prev_name, name))
        
    def _NewViewPoseEstimation(self, name): 
        
        def _Find2D3DMatches(): 
            
            matcher_temp = getattr(cv2, opts.matcher)()
            kps, descs = [], []
            for n in self.image_names: 
                if n in self.image_data.keys():
                    kp, desc = self._LoadFeatures(n)

                    kps.append(kp)
                    descs.append(desc)
            
            matcher_temp.add(descs)
            matcher_temp.train()

            kp, desc = self._LoadFeatures(name)

            matches_2d3d = matcher_temp.match(queryDescriptors=desc)

            #retrieving 2d and 3d points
            pts3d, pts2d = np.zeros((0,3)), np.zeros((0,2))
            for m in matches_2d3d: 
                train_img_idx, desc_idx, new_img_idx = m.imgIdx, m.trainIdx, m.queryIdx
                point_cloud_idx = self.image_data[self.image_names[train_img_idx]][-1][desc_idx]
                
                #if the match corresponds to a point in 3d point cloud
                if point_cloud_idx >= 0: 
                    new_pt = self.point_cloud[int(point_cloud_idx)]
                    pts3d = np.concatenate((pts3d, new_pt[np.newaxis]),axis=0)

                    new_pt = np.array(kp[int(new_img_idx)].pt)
                    pts2d = np.concatenate((pts2d, new_pt[np.newaxis]),axis=0)

            return pts3d, pts2d, len(kp)
        
        def __Find2D3DMatches():
            pts3d, pts2d = np.zeros((0,3)), np.zeros((0,2))
            kp, desc = self._LoadFeatures(name)

            i = 0 
            
            while i < len(self.image_names): 
                curr_name = self.image_names[i]

                if curr_name in self.image_data.keys(): 
                    matches = self._LoadMatches(curr_name, name)

                    ref = self.image_data[curr_name][-1]
                    pts3d_idx = np.array([ref[m.queryIdx] for m in matches \
                                        if ref[m.queryIdx] > 0])
                    pts2d_ = np.array([kp[m.trainIdx].pt for m in matches \
                                        if ref[m.queryIdx] > 0])
                                        
                    pts3d = np.concatenate((pts3d, self.point_cloud[pts3d_idx.astype(int)]),axis=0)
                    pts2d = np.concatenate((pts2d, pts2d_),axis=0)

                i += 1 

            return pts3d, pts2d, len(kp)

        pts3d, pts2d, ref_len = _Find2D3DMatches()
        _, R, t, _ = cv2.solvePnPRansac(pts3d[:,np.newaxis],pts2d[:,np.newaxis],self.K,None,
                            confidence=self.opts.pnp_prob,flags=getattr(cv2,self.opts.pnp_method),
                            reprojectionError=self.opts.reprojection_thres)
        R,_=cv2.Rodrigues(R)
        self.image_data[name] = [R,t,np.ones((ref_len,))*-1]

    def ToPly(self, filename):
        
        def _GetColors(): 
            colors = np.zeros_like(self.point_cloud)
            
            for k in self.image_data.keys(): 
                _, _, ref = self.image_data[k]
                kp, desc = self._LoadFeatures(k)
                kp = np.array(kp)[ref>=0]
                image_pts = np.array([_kp.pt for _kp in kp])

                image = cv2.imread(os.path.join(self.images_dir, k+'.jpg'))[:,:,::-1]

                colors[ref[ref>=0].astype(int)] = image[image_pts[:,1].astype(int),
                                                        image_pts[:,0].astype(int)]
            
            return colors

        colors = _GetColors()
        pts2ply(self.point_cloud, colors, filename)

    def _ComputeReprojectionError(self, name): 
        
        def _ComputeReprojections(X,R,t,K): 
            outh = K.dot(R.dot(X.T) + t )
            out = cv2.convertPointsFromHomogeneous(outh.T)[:,0,:]
            return out 

        R, t, ref = self.image_data[name]
        points_mask = ref > 0
        if not np.any(points_mask):
            print(f"Warning: No points to reproject for camera {name}")
            return 0.0
        
        reproj_pts = _ComputeReprojections(self.point_cloud[ref[ref>0].astype(int)], R, t, self.K)

        kp, desc = self._LoadFeatures(name)
        img_pts = np.array([kp_.pt for i, kp_ in enumerate(kp) if ref[i] > 0])
        
        err = np.mean(np.sqrt(np.sum((img_pts-reproj_pts)**2,axis=-1)))

        if self.opts.plot_error: 
            try:
                fig, ax = plt.subplots()
                image = cv2.imread(os.path.join(self.images_dir, name+'.jpg'))[:,:,::-1]
                ax = DrawCorrespondences(image, img_pts, reproj_pts, ax)
                
                ax.set_title(f'reprojection error = {err:.3f}')

                fig.savefig(os.path.join(self.out_err_dir, '{}.png'.format(name)))
                plt.close(fig)
            except Exception as e:
                print(f"Warning: Error plotting reprojection error for {name}: {e}")
            
        return err
        
    def Run(self):
        name1, name2 = self.image_names[0], self.image_names[1]

        total_time, errors = 0, []

        t1 = time()
        self._BaselinePoseEstimation(name1, name2)
        t2 = time()
        this_time = t2-t1
        total_time += this_time
        print('Baseline Cameras {0}, {1}: Pose Estimation [time={2:.3}s]'.format(name1, name2, this_time))

        self._TriangulateTwoViews(name1, name2)
        t1 = time()
        this_time = t1-t2
        total_time += this_time
        print('Baseline Cameras {0}, {1}: Baseline Triangulation [time={2:.3}s]'.format(name1, name2, this_time))

        views_done = 2 

        # 3d point cloud generation
        self.ToPly(os.path.join(self.out_cloud_dir, 'cloud_{}_view.ply'.format(views_done)))

        # Skip reprojection error for custom datasets
        if not self.opts.skip_reprojection:
            err1 = self._ComputeReprojectionError(name1)
            err2 = self._ComputeReprojectionError(name2)
            errors.append(err1)
            errors.append(err2)

            print('Camera {}: Reprojection Error = {}'.format(name1, err1))
            print('Camera {}: Reprojection Error = {}'.format(name2, err2))

        for new_name in self.image_names[2:]: 
            # new camera registration
            t1 = time()
            self._NewViewPoseEstimation(new_name)
            t2 = time()
            this_time = t2-t1
            total_time += this_time
            print('Camera {0}: Pose Estimation [time={1:.3}s]'.format(new_name, this_time))

            # triangulation for new registered camera
            self._TriangulateNewView(new_name)
            t1 = time()
            this_time = t1-t2
            total_time += this_time
            print('Camera {0}: Triangulation [time={1:.3}s]'.format(new_name, this_time))

            # 3d point cloud update
            views_done += 1 
            self.ToPly(os.path.join(self.out_cloud_dir, 'cloud_{}_view.ply'.format(views_done)))

            # Skip reprojection error for custom datasets
            if not self.opts.skip_reprojection:
                new_err = self._ComputeReprojectionError(new_name)
                errors.append(new_err)
                print('Camera {}: Reprojection Error = {}'.format(new_name, new_err))

        # Output final results
        if not self.opts.skip_reprojection:
            mean_error = sum(errors) / float(len(errors))
            print('Reconstruction Completed: Mean Reprojection Error = {2} [t={0:.6}s], \
                    Results stored in {1}'.format(total_time, self.opts.out_dir, mean_error))
        else:
            print('Reconstruction Completed: [t={0:.6}s], \
                    Results stored in {1}'.format(total_time, self.opts.out_dir))
        

def SetArguments(parser): 

    #directory stuff
    parser.add_argument('--data_dir',action='store',type=str,default='../data/',dest='data_dir',
                        help='root directory containing input data (default: ../data/)') 
    parser.add_argument('--dataset',action='store',type=str,default='fountain-P11',dest='dataset',
                        help='name of dataset (default: fountain-P11)') 
    parser.add_argument('--ext',action='store',type=str,default='jpg,png',dest='ext', 
                        help='comma seperated string of allowed image extensions \
                        (default: jpg,png)') 
    parser.add_argument('--out_dir',action='store',type=str,default='../results/',dest='out_dir',
                        help='root directory to store results in (default: ../results/)') 

    #matching parameters
    parser.add_argument('--features',action='store', type=str, default='ORB', dest='features',
                        help='[SIFT|SURF|ORB] Feature algorithm to use (default: ORB)') 
    parser.add_argument('--matcher',action='store',type=str,default='BFMatcher',dest='matcher',
                        help='[BFMatcher|FlannBasedMatcher] Matching algorithm to use \
                        (default: BFMatcher)') 
    parser.add_argument('--cross_check',action='store',type=bool,default=True,dest='cross_check',
                        help='[True|False] Whether to cross check feature matching or not \
                        (default: True)') 

    #epipolar geometry parameters
    parser.add_argument('--calibration_mat',action='store',type=str,default='benchmark',
                        dest='calibration_mat',help='[benchmark|lg_g3] type of intrinsic camera \
                        to use (default: benchmark)')
    parser.add_argument('--fund_method',action='store',type=str,default='FM_RANSAC',
                        dest='fund_method',help='method to estimate fundamental matrix \
                        (default: FM_RANSAC)')
    parser.add_argument('--outlier_thres',action='store',type=float,default=.9,
                        dest='outlier_thres',help='threhold value of outlier to be used in\
                         fundamental matrix estimation (default: 0.9)')
    parser.add_argument('--fund_prob',action='store',type=float,default=.9,dest='fund_prob',
                        help='confidence in fundamental matrix estimation required (default: 0.9)')
    
    #PnP parameters
    parser.add_argument('--pnp_method',action='store',type=str,default='SOLVEPNP_DLS',
                        dest='pnp_method',help='[SOLVEPNP_DLS|SOLVEPNP_EPNP|..] method used for\
                        PnP estimation, see OpenCV doc for more options (default: SOLVEPNP_DLS')
    parser.add_argument('--pnp_prob',action='store',type=float,default=.99,dest='pnp_prob',
                        help='confidence in PnP estimation required (default: 0.99)')
    parser.add_argument('--reprojection_thres',action='store',type=float,default=8.,
                        dest='reprojection_thres',help='reprojection threshold in PnP estimation \
                        (default: 8.)')

    #misc
    parser.add_argument('--plot_error',action='store',type=bool,default=False,dest='plot_error')

    # Add these arguments to the SetArguments function
    parser.add_argument('--out_cloud_dir', action='store', type=str, default='', 
                        dest='out_cloud_dir', help='custom directory to store point cloud results')
    parser.add_argument('--out_err_dir', action='store', type=str, default='',
                        dest='out_err_dir', help='custom directory to store error visualizations')

    parser.add_argument('--feat_in_dir', action='store', type=str, default='', 
                        dest='feat_in_dir', help='custom directory to read feature files from')
    parser.add_argument('--matches_in_dir', action='store', type=str, default='',
                        dest='matches_in_dir', help='custom directory to read matches files from')
    
    # Add parameters for custom uploads
    parser.add_argument('--custom_intrinsics', action='store', type=bool, default=False, 
                        dest='custom_intrinsics', help='Use custom intrinsics matrix from calibration file')
    parser.add_argument('--skip_reprojection', action='store', type=bool, default=False, 
                        dest='skip_reprojection', help='Skip reprojection error calculation')

def PostprocessArgs(opts): 
    opts.fund_method = getattr(cv2,opts.fund_method)
    opts.ext = opts.ext.split(',')

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    SetArguments(parser)
    opts = parser.parse_args()
    PostprocessArgs(opts)
    
    sfm = SFM(opts)
    sfm.Run()
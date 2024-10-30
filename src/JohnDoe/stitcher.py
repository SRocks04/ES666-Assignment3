import pdb
import numpy as np
import glob
import cv2
import os
from src.JohnDoe.some_function import *

def Homography_opencv(correspondences, trials=1000, threshold=10, num_samples=4):
    srcPoints = np.float32([point[0] for point in correspondences])
    dstPoints = np.float32([point[1] for point in correspondences])
    H, mask = cv2.findHomography(srcPoints, dstPoints, method=cv2.RANSAC, ransacReprojThreshold=threshold)
    return H, mask

class PanaromaStitcher():
    def __init__(self):
        pass

    def stitch_and_save_images_opencv(self, src_idx, dest_idx, prev_homography):
        '''
        For a given pair of image indices, computes the best homography 
        matrix and saves the warped images to disk.

        Args:
            src_idx: Index of the source image in imagePaths list.
            dest_idx: Index of the destination image in imagePaths list.
            prev_homography: Previous cumulative homography matrix.

        Returns:
            new_cumulative_homography: Updated cumulative homography matrix.
        '''
        
        # Initialize the destination warped image with zeros
        warped_image = np.zeros((3000, 6000, 3), dtype=np.uint8)

        # Read source and destination images
        src_img = cv2.imread(self.all_images[src_idx])
        dest_img = cv2.imread(self.all_images[dest_idx])

        print(f'Original image size = {src_img.shape}')

        # Resize images for performance and compatibility
        resized_src_img = cv2.resize(src_img, self.shape)
        resized_dest_img = cv2.resize(dest_img, self.shape)

        # Detect features and perform matching between images
        matches, src_pts, dest_pts = detectFeaturesAndMatch(resized_dest_img, resized_src_img)

        # Compute the best homography using RANSAC
        best_homography, _ = Homography_opencv(matches, trials=self.trials, threshold=self.threshold)

        # Update the cumulative homography matrix
        new_cumulative_homography = np.dot(prev_homography, best_homography)

        # Transform the source image and place it in the warped image
        warpAndPlaceSourceImage(resized_dest_img, new_cumulative_homography, dest_img=warped_image, offset=self.offset)


        return warped_image,new_cumulative_homography

    def make_panaroma_for_images_in(self,path):
        imf = path
        self.all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(self.all_images)))


        # Collect all homographies calculated for pair of images and return
        homography_matrix_list =[]
        self.shape = (600, 400)
        mid = len(self.all_images)//2
        self.threshold = 2
        self.trials = 3000
        self.offset = [2300, 800]
        Hom = np.eye(3)
        b = ImageBlenderWithPyramids()
        # Return Final panaroma
        stitched_image = cv2.imread(self.all_images[0])
        stitched_image, Hom = self.stitch_and_save_images_opencv(1, 0, Hom)
        print(0)
        homography_matrix_list.append(Hom)
        for i in range(2,len(self.all_images)-1):
            warp,Hom=self.stitch_and_save_images_opencv(i, i-1, Hom)
            print(i-1)
            homography_matrix_list.append(Hom)
            stitched_image = b.blendImages(stitched_image, warp)
        Hom = np.eye(3)
        warp,Hom=self.stitch_and_save_images_opencv(-2, -2, Hom)
        stitched_image = b.blendImages(stitched_image, warp)
        homography_matrix_list.append(Hom)
        warp,Hom=self.stitch_and_save_images_opencv(-2, -1, Hom)
        stitched_image = b.blendImages(stitched_image, warp)
        homography_matrix_list.append(Hom)
        #####
        return stitched_image, homography_matrix_list 

    
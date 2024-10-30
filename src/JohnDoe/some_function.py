import cv2
import numpy as np

def detectFeaturesAndMatch(image1, image2, maxNumOfFeatures=30):
    '''
    Takes two images as input and returns a set of correspondences between the two images, matched using SIFT features.
    The matches are sorted from best to worst based on the Euclidean (L2) distance between feature descriptors.
    '''

    # Initialize the SIFT feature detector and descriptor
    siftDetector = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = siftDetector.detectAndCompute(image1, None)
    keypoints2, descriptors2 = siftDetector.detectAndCompute(image2, None)

    # Initialize BFMatcher with L2 norm for matching SIFT descriptors
    bruteForceMatcher = cv2.BFMatcher(cv2.NORM_L2)

    # Perform the matching between descriptors of two images
    rawMatches = bruteForceMatcher.match(descriptors1, descriptors2)

    # Sort the matches based on their distance attributes
    sortedMatches = sorted(rawMatches, key=lambda x: x.distance)

    # Initialize an empty list to store corresponding points between two images
    featureCorrespondences = []
    for match in sortedMatches:
        featureCorrespondences.append((keypoints1[match.queryIdx].pt, keypoints2[match.trainIdx].pt))

    print(f'Total number of matches: {len(featureCorrespondences)}')

    # Prepare source and destination points for homography
    sourcePoints = np.float32([point_pair[0] for point_pair in featureCorrespondences[:maxNumOfFeatures]]).reshape(-1, 1, 2)
    destinationPoints = np.float32([point_pair[1] for point_pair in featureCorrespondences[:maxNumOfFeatures]]).reshape(-1, 1, 2)

    return np.array(featureCorrespondences[:maxNumOfFeatures]), sourcePoints, destinationPoints

class ImageBlenderWithPyramids():
    '''
    Class for blending images using Gaussian and Laplacian pyramids.
    It is specialized for the 'STRAIGHTCUT' blending strategy.
    '''

    def __init__(self, pyramid_depth=6):
        '''
        Initialize with the depth of the pyramids.
        '''
        self.pyramid_depth = pyramid_depth

    def getGaussianPyramid(self, image):
        '''
        Generate a Gaussian pyramid for a given image.
        '''
        pyramid = [image]
        for _ in range(self.pyramid_depth - 1):
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid

    def getLaplacianPyramid(self, image):
        '''
        Generate a Laplacian pyramid for a given image.
        '''
        pyramid = []
        for _ in range(self.pyramid_depth - 1):
            next_level_image = cv2.pyrDown(image)
            upsampled_image = cv2.pyrUp(next_level_image, dstsize=(image.shape[1], image.shape[0]))
            pyramid.append(image.astype(float) - upsampled_image.astype(float))
            image = next_level_image
        pyramid.append(image)
        return pyramid

    def getBlendingPyramid(self, laplacian_a, laplacian_b, gaussian_mask_pyramid):
        '''
        Create a blended Laplacian pyramid using two Laplacian pyramids and a Gaussian pyramid as mask.
        '''
        blended_pyramid = []
        for i, mask in enumerate(gaussian_mask_pyramid):
            triplet_mask = cv2.merge((mask, mask, mask))
            blended_pyramid.append(laplacian_a[i] * triplet_mask + laplacian_b[i] * (1 - triplet_mask))
        return blended_pyramid

    def reconstructFromPyramid(self, laplacian_pyramid):
        '''
        Reconstruct an image from its Laplacian pyramid.
        '''
        reconstructed_image = laplacian_pyramid[-1]
        for laplacian_level in reversed(laplacian_pyramid[:-1]):
            reconstructed_image = cv2.pyrUp(reconstructed_image, dstsize=laplacian_level.shape[:2][::-1]).astype(float) + laplacian_level.astype(float)
        return reconstructed_image

    def generateMaskFromImage(self, image):
        '''
        Generate a mask based on the non-zero regions of the image.
        '''
        mask = np.all(image != 0, axis=2)
        mask_image = np.zeros(image.shape[:2], dtype=float)
        mask_image[mask] = 1.0
        return mask_image

    def blendImages(self, image1, image2):
        '''
        Blend two images using Laplacian pyramids and a Gaussian mask.
        '''
        laplacian1 = self.getLaplacianPyramid(image1)
        laplacian2 = self.getLaplacianPyramid(image2)

        mask1 = self.generateMaskFromImage(image1).astype(np.bool_)
        mask2 = self.generateMaskFromImage(image2).astype(np.bool_)
        
        overlap_region = mask1 & mask2

        y_coords, x_coords = np.where(overlap_region)
        min_x, max_x = np.min(x_coords), np.max(x_coords)

        final_mask = np.zeros(image1.shape[:2])
        final_mask[:, :(min_x + max_x)//2] = 1.0

        gaussian_mask_pyramid = self.getGaussianPyramid(final_mask)

        blended_pyramid = self.getBlendingPyramid(laplacian1, laplacian2, gaussian_mask_pyramid)

        blended_image = self.reconstructFromPyramid(blended_pyramid)

        return blended_image

def computeBoundingBoxOfWarpedImage(homography_matrix, img_width, img_height):
    """
    Compute the bounding box of an image after it has been transformed by a given homography matrix.

    Args:
        homography_matrix (np.ndarray): 3x3 matrix used for the homographic transformation.
        img_width (int): Width of the original image.
        img_height (int): Height of the original image.

    Returns:
        tuple: Coordinates of the bounding box as (x_min, x_max, y_min, y_max).
    """
    
    # Define corners of the original image [Top-left, Top-right, Bottom-left, Bottom-right]
    # Each column is a corner point in homogeneous coordinates (x, y, 1)
    original_corners = np.array([[0, img_width - 1, 0, img_width - 1], 
                                 [0, 0, img_height - 1, img_height - 1], 
                                 [1, 1, 1, 1]])
    
    # Apply the homography to the original corners
    transformed_corners = np.dot(homography_matrix, original_corners)
    
    # Convert to inhomogeneous coordinates by dividing by the third (w) component
    transformed_corners /= transformed_corners[2, :]
    
    # Find the minimum and maximum x and y coordinates of the transformed corners
    x_min = np.min(transformed_corners[0])
    x_max = np.max(transformed_corners[0])
    y_min = np.min(transformed_corners[1])
    y_max = np.max(transformed_corners[1])
    
    return int(x_min), int(x_max), int(y_min), int(y_max)

def warpAndPlaceSourceImage(source_img, homography_matrix, dest_img, offset=(0, 0)):
    """
    Warps the source image according to a given homography matrix and places it onto the destination image.

    Args:
        source_img (np.ndarray): The source image array to be warped.
        homography_matrix (np.ndarray): The 3x3 matrix governing the homographic transformation.
        dest_img (np.ndarray): The destination image array where the warped source image will be placed.
        use_forward_mapping (bool): Determines whether to use forward or inverse mapping for the transformation.
        offset (tuple): Offsets for placing the warped image onto the destination image in (x, y) coordinates.

    Returns:
        None: The destination image is modified in place.
    """

    # Extract the dimensions of the source image
    height, width, _ = source_img.shape
    
    # Compute the inverse of the homography matrix for inverse mapping
    homography_inv = np.linalg.inv(homography_matrix)

    x_min, x_max, y_min, y_max = computeBoundingBoxOfWarpedImage(homography_matrix, width, height)
    
    # Generate indices for destination image coordinates within bounding box
    coords = np.indices((x_max - x_min, y_max - y_min)).reshape(2, -1)
    coords += np.array([[x_min], [y_min]])
    homogeneous_coords = np.vstack((coords, np.ones(coords.shape[1])))
    
    # Apply the inverse homography transformation
    transformed_coords = np.dot(homography_inv, homogeneous_coords)
    transformed_coords /= transformed_coords[2, :]
    
    # Extract the integer coordinates of the input
    x_input, y_input = transformed_coords.astype(np.int32)[:2, :]
    
    # Perform boundary check to make sure coordinates are within source image dimensions
    valid_indices = np.where((x_input >= 0) & (x_input < width) & (y_input >= 0) & (y_input < height))

    # Map the inverse-transformed points to the source image and place onto the destination image
    dest_img[coords[1, valid_indices] + offset[1], coords[0, valid_indices] + offset[0]] = source_img[y_input[valid_indices], x_input[valid_indices]]

import cv2
import glob
import numpy as np
from skimage.feature import hog


def load_image(image_dir):
    image_names = glob.glob(image_dir+"*")
    images = []
    for image_name in image_names:
        img = cv2.imread(image_name,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (256, 256))
        images.append(img)
    return np.array(images)

def cluster_image(image):
    vectorized_image=image.reshape((-1,3))
    vectorized_image=np.float32(vectorized_image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) # interation termination criteria
    ret,labels,centers=cv2.kmeans(vectorized_image,5,None,criteria,10,cv2.KMEANS_PP_CENTERS)
    centers=np.uint8(centers)
    res=centers[labels.flatten()]
    result_image=res.reshape((image.shape))
    return result_image
def HOG(images):
    features = []
    for idx, image in enumerate(images):
        # clustered_image=cluster_image(image)
        fd, hog_image = hog(image, orientations=9, pixels_per_cell=(
            8, 8), cells_per_block=(3, 3), visualize=True)
        features.append(fd)
        print('[%d/%d] Done HOG feature extraction' % (idx+1, len(images)))
    return np.array(features)

import os
import cv2
import scipy
import scipy.ndimage
from tqdm import tqdm

# 设置当前工作目录为项目根目录
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
video_path = os.path.join(os.getcwd(), "assets", "videos", "video1.mp4")

# argument param
brightness = 60
rgb_gaussian_sigma = 20
gaussian_blur = 11


def extended_data(ori_pic):
    '''
    argument pic
    @param ori_pic: origin picture
    @return: a dictionary of processed data
    '''
    pics = {}
    # brightness
    pic_brighter = ori_pic.copy()
    pic_brighter = cv2.add(pic_brighter, brightness)
    # rgb jitter
    pic_noisy = ori_pic.copy()
    for i in range(3):
        pic_noisy[:, :, i] = scipy.ndimage.gaussian_filter(
            pic_noisy[:,:,i], sigma=rgb_gaussian_sigma)
    # blur
    pic_blur = cv2.GaussianBlur(
        ori_pic, (gaussian_blur, gaussian_blur), 0)
    
    pics["brighter"] = pic_brighter
    pics["jitter"] = pic_noisy
    pics["blur"] = pic_blur
    
    return pics
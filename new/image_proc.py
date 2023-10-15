if __name__ == '__main__':
    import sys
    from PyQt5 import QtWidgets
    QtWidgets.QApplication(sys.argv)


import numpy as np
import cv2
import function as f
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import math

def scurve(x, a):
    # 3 order polynomial passing through (0,0), (0.5,0.5), (1,1)
    # 0<=x<=1
    # 2>=a>=-2

    if x < 0:
        return 0
    elif x > 1:
        return 1
    else:
        d = 0
        b = a*(-3/2)
        c = 1+(0.5*a)
        return a*(x**3) + b*(x**2) + c*x + d
    
def normalize(img):
    img = img/np.max(img)
    return img


def contrast(img, a):
    # -1<=a<=1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = scurve(img[i,j], a*-2)
    return img


def binarize(img, threshold):
    img[img < threshold] = 0
    img[img >= threshold] = 1
    return img

def barcode_detection(img, dilate_size, erode_size, morph_iter):
    # apply sobel filter using convolution
    print('Applying sobel filter...')
    
    sobel_h = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    sobel_v = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    # apply convolution using numpy
    # img_h = np.zeros(img.shape)
    # img_v = np.zeros(img.shape)
    # img = np.pad(img, pad_width=1, mode='constant', constant_values=0)
    # for i in tqdm(range(1, img.shape[0]-1)):
    #     for j in range(1, img.shape[1]-1):
    #         img_h[i,j] = np.sum(img[i-1:i+2, j-1:j+2]*sobel_h)
    #         img_v[i,j] = np.sum(img[i-1:i+2, j-1:j+2]*sobel_v)

    # apply convolution using pytorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sobel_h = torch.from_numpy(sobel_h).float().to(device)
    sobel_v = torch.from_numpy(sobel_v).float().to(device)
    img = torch.from_numpy(img).float().to(device)

    img = img.unsqueeze(0).unsqueeze(0)
    sobel_h = sobel_h.unsqueeze(0).unsqueeze(0)
    sobel_v = sobel_v.unsqueeze(0).unsqueeze(0)

    img_h = torch.nn.functional.conv2d(img, sobel_h, padding=1)
    img_v = torch.nn.functional.conv2d(img, sobel_v, padding=1)

    img_h = img_h.squeeze(0).squeeze(0).cpu().numpy()
    img_v = img_v.squeeze(0).squeeze(0).cpu().numpy()

    # calculate gradient magnitude
    img_mag = np.sqrt(img_h**2 + img_v**2)
    # img_mag = normalize(img_mag)


    # binarize
    img_mag = binarize(img_mag, 1)

    # morphing
    print('Applying morphing...')
    for i in tqdm(range(morph_iter)):
        if dilate_size > 0:
            img_mag = dilate(img_mag, dilate_size)
            img_mag = erode(img_mag, erode_size)
    # for i in tqdm(range(morph_iter)):
    #     if erode_size > 0:
    #         img_mag = erode(img_mag, erode_size)
    
    # img_mag = 1 - img_mag
    
    # object detection
    print('Detecting objects...')
    label_table = f.get_object(img_mag)
    obj_num = label_table.obj_num
    label_table_int = label_table.get_int_array()
    max_area_obj = np.argmax(label_table.get_area())
    max_area = label_table.get_area()[max_area_obj]
    max_area_mask = label_table.get_obj_mask(max_area_obj)
    location = label_table.get_location()[max_area_obj]

    # make an rectangle bounding box
    print('Making bounding box...')
    # max_area_mask = max_area_mask.astype(np.uint8)
    # contours, hierarchy = cv2.findContours(max_area_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return img_mag, max_area_mask, location


def dilate(img, kernel_size):
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    border = kernel_size//2
    height, width = img.shape
    paddedIm = np.pad(img, (border, border), 'constant', constant_values=(0, 0))
    paddedDilatedIm = paddedIm.copy()

    for h_i in range(border, height+border):
        for w_i in range(border,width+border):
            # When you find a white pixel
            if img[h_i-border,w_i-border]:
                # print("White Pixel Found @ {},{}".format(h_i,w_i))
                
                paddedDilatedIm[ h_i - border : (h_i + border)+1, w_i - border : (w_i + border)+1] = np.logical_or(paddedDilatedIm[ h_i - border : (h_i + border)+1, w_i - border : (w_i + border)+1], element)
                                
    dilatedImage = paddedDilatedIm[border:border+height,border:border+width]

    return dilatedImage

def erode(img, kernel_size):
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    border = kernel_size//2
    height, width = img.shape
    paddedIm = np.pad(img, (border, border), 'constant', constant_values=(0, 0))
    paddedErodedIm = paddedIm.copy()
    paddedErodedIm2= paddedIm.copy()

    roi=0
    temp=0
    for h_i in range(border, height+border):
        for w_i in range(border,width+border):
            if img[h_i-border,w_i-border]:
                roi=paddedErodedIm2[h_i-border  : (h_i + border)+1, w_i - border : (w_i + border)+1] 
                temp = np.logical_and(roi,element)
                paddedErodedIm[h_i,w_i]=np.min(temp)


    erodedImage = paddedErodedIm[border:border+height,border:border+width]

    return erodedImage

def median_blur(img, ksize):
    if ksize % 2 == 0:
        ksize += 1
    img = np.pad(img, pad_width=ksize//2, mode='constant', constant_values=0)
    for i in range(ksize//2, img.shape[0]-ksize//2):
        for j in range(ksize//2, img.shape[1]-ksize//2):
            img[i,j] = np.median(img[i-ksize//2:i+ksize//2+1, j-ksize//2:j+ksize//2+1])
    return img[ksize//2:-ksize//2, ksize//2:-ksize//2]

def gaussian_blur(img):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]])
    kernel = kernel/np.sum(kernel)

    img = torch.from_numpy(img).float().to(device)
    kernel = torch.from_numpy(kernel).float().to(device)

    img = img.unsqueeze(0).unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    img_blur = torch.nn.functional.conv2d(img, kernel, padding=1)
    img_blur = img_blur.squeeze(0).squeeze(0).cpu().numpy()
    return img_blur


def adaptive_binarize(img, ksize=(100, 100), th_shift=0):
    H, W = img.shape

    # split image into 100x100 blocks
    image_new = np.zeros((H, W))
    for i in range(0, H, ksize[0]):
        for j in range(0, W, ksize[1]):
            th = get_best_threshold(img[i:i+ksize[0], j:j+ksize[1]])
            th += th_shift
            image_new[i:i+ksize[0], j:j+ksize[1]] = binarize(img[i:i+ksize[0], j:j+ksize[1]], th)

    return image_new

def get_best_threshold(img):
    resolution = 100
    img_px = img.reshape(-1)
    hist = np.zeros(resolution)
    hist_tmp = 0
    for i in range(resolution):
        # hist[i] is the number of pixels whose value is in i/100 and (i+1)/100
        if i == 99:
            val = np.where((img_px >= i/resolution) & (img_px <= (i+1)/resolution))[0]
        else:
            val = np.where((img_px >= i/resolution) & (img_px < (i+1)/resolution))[0]
        
        hist[i] = np.sum(img_px[val])

    # plt.plot(hist)
    # plt.show()

    _, hist_mass_center = get_mass_center(hist, 0, resolution)
    last_th = hist_mass_center

    while True:
        _, hist_mass_center_left = get_mass_center(hist, 0, int(last_th))
        _, hist_mass_center_right = get_mass_center(hist, int(last_th), resolution)
        hist_mass_center = (hist_mass_center_left + hist_mass_center_right)/2
        if np.abs(hist_mass_center - last_th) < 0.01:
            return hist_mass_center/resolution
        else:
            last_th = hist_mass_center


def get_mass_center(f, a, b):
    # return integral(i*f,a,b)/integral(f,a,b)
    area = 0
    area_tmp = 0
    for i in range(a, b):
        area += f[i]
        area_tmp += i*f[i]
    
    return area, area_tmp/area

def bilinear_interpolation(image, y, x):
    height = image.shape[0]
    width = image.shape[1]

    x1 = max(min(math.floor(x), width - 1), 0)
    y1 = max(min(math.floor(y), height - 1), 0)
    x2 = max(min(math.ceil(x), width - 1), 0)
    y2 = max(min(math.ceil(y), height - 1), 0)

    a = float(image[y1, x1])
    b = float(image[y2, x1])
    c = float(image[y1, x2])
    d = float(image[y2, x2])

    dx = x - x1
    dy = y - y1

    new_pixel = a * (1 - dx) * (1 - dy)
    new_pixel += b * dy * (1 - dx)
    new_pixel += c * dx * (1 - dy)
    new_pixel += d * dx * dy
    return round(new_pixel)


def resize(image, size):
    new_height, new_width = size
    new_image = np.zeros((new_height, new_width), image.dtype)  # new_image = [[0 for _ in range(new_width)] for _ in range(new_height)]

    orig_height = image.shape[0]
    orig_width = image.shape[1]

    # Compute center column and center row
    x_orig_center = (orig_width-1) / 2
    y_orig_center = (orig_height-1) / 2

    # Compute center of resized image
    x_scaled_center = (new_width-1) / 2
    y_scaled_center = (new_height-1) / 2

    # Compute the scale in both axes
    scale_x = orig_width / new_width;
    scale_y = orig_height / new_height;

    for y in range(new_height):
        for x in range(new_width):
            x_ = (x - x_scaled_center) * scale_x + x_orig_center
            y_ = (y - y_scaled_center) * scale_y + y_orig_center

            new_image[y, x] = bilinear_interpolation(image, y_, x_)

    return new_image
import numpy as np
import cv2
import function as f
from tqdm import tqdm
import torch

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
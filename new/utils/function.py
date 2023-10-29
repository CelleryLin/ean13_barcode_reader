import numpy as np
import cv2
from utils.objlabels import labelNode, labelTable
import matplotlib.pyplot as plt
import time

def get_object(img):
    # Sequential labeling algorithm
    paddedIM = np.pad(img, pad_width=1, mode='constant', constant_values=0)
    label_table = np.zeros(paddedIM.shape, dtype=object)

    for i in range(label_table.shape[0]):
        for j in range(label_table.shape[1]):
            label_table[i,j] = None

    label = 0
    for i in range(1, paddedIM.shape[0]-1):
        for j in range(1, paddedIM.shape[1]-1):
            if paddedIM[i,j] == 1:
                
                # both upper and left are not labeled
                if label_table[i-1,j] is None and label_table[i,j-1] is None:
                    label += 1
                    label_table[i,j] = labelNode(label)
                
                # one of upper or left is not labeled (2 situations)
                elif label_table[i-1,j] is not None and label_table[i,j-1] is None:
                    label_table[i,j] = label_table[i-1,j]

                elif label_table[i-1,j] is None and label_table[i,j-1] is not None:
                    label_table[i,j] = label_table[i,j-1]

                # both upper and left are labeled (2 situations)
                elif label_table[i-1,j].get_label() == label_table[i,j-1].get_label():
                    label_table[i,j] = label_table[i-1,j]
                
                elif label_table[i-1,j].get_label() != label_table[i,j-1].get_label():
                    label_table[i,j] = label_table[i-1,j]
                    label_table[i,j-1].insert(label_table[i,j])

                else:
                    raise Exception('unknown error')
    
    # find the root
    for i in range(label_table.shape[0]):
        for j in range(label_table.shape[1]):
            if label_table[i,j] is not None:
                label_table[i,j] = label_table[i,j].get_root()

    # find the number of objects
    label_table = labelTable(label_table[1:-1,1:-1])


    return label_table

def apply_scanline(img, line_num, angle, resolution=1000):
    # apply equidistant scanline in img size with given angle
    # img: input image
    # line_num: number of scanlines
    # angle: angle of scanlines (degree)

    # current_time = time.time()
    H, W = img.shape
    midpoint = ((W-1)/2, (H-1)/2)
    angle = np.deg2rad(angle)
    # y = tan(angle) * (x-a) + b
    diag_angle = (np.arctan2(H, W), np.arctan2(-H, W))
    # equidistant points on main line
    if angle <= diag_angle[0] and angle >= diag_angle[1]:
        # ends of main line are lies on the left and right side of image
        x = np.linspace(0, W-1, line_num)
        y = np.tan(angle) * (x - midpoint[0]) + midpoint[1]
    else:
        # ends of main line are lies on the top and bottom side of image
        y = np.linspace(0, H-1, line_num)
        x = (y - midpoint[1]) / np.tan(angle) + midpoint[0]

    lines = []
    angle_ = angle - np.pi/2
    for x_, y_ in zip(x, y):
        px_val = []
        # y = tan(angle) * (x-a) + b
        # a is x_, b is y_, angle is angle-(pi/2)
        if angle_ <= diag_angle[0] and angle_ >= diag_angle[1]:
            i_vals = np.linspace(0, W - 1, resolution)
            j_vals = np.tan(angle_) * (i_vals - x_) + y_
            mask = (i_vals >= 0) & (i_vals <= W-1) & (j_vals >= 0) & (j_vals <= H-1)

        else:
            j_vals = np.linspace(0, H - 1, resolution)
            i_vals = (j_vals - y_) / np.tan(angle_) + x_
            mask = (i_vals >= 0) & (i_vals <= W-1) & (j_vals >= 0) & (j_vals <= H-1)
        
        indices = j_vals[mask].astype(int), i_vals[mask].astype(int)
        px_val = img[indices]
        lines.append(px_val)

    # time_last = time.time()-current_time
    # print('scanline time: ', time_last)

    return x,y,lines

def plot_scanline(img, x, y, angle):
    for x_, y_ in zip(x, y):
        xx = np.linspace(0, img.shape[1], 100)
        yy = y_ + (xx - x_) * np.tan((angle-90) / 180 * np.pi)
        plt.plot(xx, yy, c="r")
    plt.imshow(img, cmap="gray")
    plt.show()



def is_in_interval(obj, n, tolerance):
    # check if obj is in interval n with tolerance
    if obj >= n-tolerance and obj < n+tolerance:
        return True
    else:
        return False
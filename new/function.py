import numpy as np
import cv2
from objlabels import labelNode, labelTable

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

def apply_scanline(img, line_num, angle, tilt=None, resolution=1000):
    # apply equidistant scanline in img size with given angle
    # img: input image
    # line_num: number of scanlines
    # angle: angle of scanlines (degree)

    H, W = img.shape
    midpoint = ((W-1)/2, (H-1)/2)
    angle = np.deg2rad(angle)
    # y = tan(angle) * (x-a) + b
    diag_angle = (np.arctan2(H, W), np.arctan2(-H, W))
    # equidistant points on main line
    if np.tan(angle) <= diag_angle[0] and np.tan(angle) >= diag_angle[1]:
        # ends of main line are lies on the left and right side of image
        x = np.linspace(0, W-1, line_num)
        y = np.tan(angle) * (x - midpoint[0]) + midpoint[1]
    else:
        # ends of main line are lies on the top and bottom side of image
        y = np.linspace(0, H-1, line_num)
        x = (y - midpoint[1]) / np.tan(angle) + midpoint[0]

    lines = []
    for x_, y_ in zip(x, y):
        px_val = []
        # y = tan(angle) * (x-a) + b
        # a is x_, b is y_, angle is angle-(pi/2)
        angle_ = angle - np.pi/2
        if np.tan(angle_) <= diag_angle[0] and np.tan(angle_) >= diag_angle[1]:
            for i in np.linspace(0, W-1, resolution):
                j = np.tan(angle_) * (i-x_) + y_
                if i < 0 or i >= W or j < 0 or j >= H:
                    continue
                px_val.append(img[int(j), int(i)])
            lines.append(px_val)

        else:
            for j in np.linspace(0, H-1, resolution):
                i = (j-y_) / np.tan(angle_) + x_
                if i < 0 or i >= W or j < 0 or j >= H:
                    continue
                px_val.append(img[int(j), int(i)])

            # tilt correction
            
            px_val_tilt = []
            if tilt is None:
                px_val_tilt = px_val

            elif tilt >= 0:
                correction_table = np.linspace(0, tilt, len(px_val))
                for i in range(len(px_val)):
                    px_val_tilt.extend([px_val[i]] * (1+int(correction_table[i])))
            
            elif tilt < 0:
                correction_table = np.linspace(tilt, 0, len(px_val))
                for i in range(len(px_val)):
                    px_val_tilt.extend([px_val[i]] * (1+int(correction_table[i])))
            
            lines.append(px_val)
    return x,y,lines

def is_in_interval(obj, n, tolerance):
    # check if obj is in interval n with tolerance
    if obj >= n-tolerance and obj < n+tolerance:
        return True
    else:
        return False
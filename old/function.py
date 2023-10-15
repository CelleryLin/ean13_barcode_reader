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


def area_calc(label_table, obj_num):
    area = np.zeros(obj_num+1)
    for i in range(label_table.shape[0]):
        for j in range(label_table.shape[1]):
            if label_table[i,j] != 0:
                area[label_table[i,j]] += 1
    return area
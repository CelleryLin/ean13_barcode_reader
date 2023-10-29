import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import utils.image_proc as ip
import utils.function as f
from utils.decode import EAN13DECODER
import time

import multiprocessing as mp

def solver(img_folder, line_num, angles, torelances, show_res=False, show_drop=False, use_mp=True):
    accuracy = 0
    all_codes = []
    all_images = []
    
    for img_path in tqdm(os.listdir(img_folder)):
        if os.path.isdir(os.path.join(img_folder, img_path)):
            continue
        curr_time = time.time()
        img = cv2.imread(os.path.join(img_folder, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = np.mean(img, axis=2).astype(np.uint8)
        img_org = img.copy()
        
        all_images.append(img_path)
        H, W = img.shape[:2]
        # if H > W:
        #     # img = cv2.resize(img, (600, 800))
        #     img = ip.resize(img, (600, 800))
        # else:
        #     # img = cv2.resize(img, (800, 600))
        #     img = ip.resize(img, (800, 600))

        if line_num == 'auto':
            line_num = np.sqrt(H**2 + W**2) / 10
            line_num = int(line_num)

        img = ip.normalize(img)        
        img = ip.gaussian_blur(img)
        img = ip.adaptive_binarize(img, th_shift=-0.1)
        img = 1 - img

        # plt.imshow(img, cmap="gray")
        # plt.show()
        

        status = 'fail'
        break_all = False
        
        for angle in angles:
            x,y,lines = f.apply_scanline(img, line_num, angle, resolution=1000)
            for torelance in torelances:
                # f.plot_scanline(img, x, y, angle)
                for i in lines:
                    if i is None:
                        continue
                    try:
                        i = np.array(i)
                        
                        decoder = EAN13DECODER(i, tolerance=torelance)
                        
                        # decoder.plot_line()
                        _, code, status = decoder.get_barcode()
                        
                        # print(code)
                        if status == 'success':
                            final_code = code
                            if show_res == True:
                                show_text = 'OK ' + final_code
                                plt.imshow(img_org, cmap="gray")
                                plt.text(5, -2, show_text, ha='left', va='bottom', fontsize=14, color="g")
                                plt.show()
                            elif show_res == 'only_code':
                                print(img_path, ": ", final_code)

                            # Success, end searching
                            accuracy += 1
                            all_codes.append(final_code)
                            break_all = True
                            break

                    except Exception as e:
                        # if is DECODERERROR, then continue
                        if e.__class__.__name__ == "DECODERERROR":
                            # print(e)
                            continue
                        else:
                            raise e
                
                if break_all:
                    break

            if break_all:
                break

        if status == 'fail':
            if show_res == True:
                show_text = 'FAIL'
                plt.imshow(img_org, cmap="gray")
                plt.text(5, -2, show_text, ha='left', va='bottom', fontsize=14, color="r")
                plt.show()
            elif show_res == 'only_code':
                print(img_path, ": fail")
            all_codes.append("fail")
            # print("Fail! Please drop this class.")
            
            if show_drop:
                import webbrowser
                webbrowser.open_new('./fail.pdf')

    avg_duration = time.time() - curr_time
    avg_duration /= len(all_images)
    return accuracy/len(all_images), avg_duration, all_codes, all_images


# def solver_once(img, line_num, angle, tilt, torelance, show_res):
#     x,y,lines = f.apply_scanline(img, line_num, angle, resolution=1000)
#     for i in lines:
#         if i is None:
#             continue
#         try:
#             i = np.array(i)
#             decoder = EAN13DECODER(i, tolerance=torelance)
#             # decoder.plot_line()
#             _, code, status = decoder.get_barcode()
#             # print(code)
#             if status == 'success':
#                 final_code = code
#                 if show_res:
#                     show_text = 'OK ' + final_code
#                     plt.imshow(img, cmap="gray")
#                     plt.text(5, -2, show_text, ha='left', va='bottom', fontsize=14, color="g")
#                     plt.show()
#                 raise DECODEDONE('done')

#         except Exception as e:
#             # if is DECODERERROR, then continue
#             if e.__class__.__name__ == "DECODERERROR":
#                 # print(e)
#                 return 'fail'
#                 continue
#             else:
#                 if e.__class__.__name__ == "DECODEDONE":
#                     # print("Done!")
#                     return final_code
#                     accuracy += 1
#                     all_codes.append(final_code)
#                     break_all = True
#                     break
#                 else:
#                     raise e
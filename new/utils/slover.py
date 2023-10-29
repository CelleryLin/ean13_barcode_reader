import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import utils.image_proc as ip
import utils.function as f
from utils.decode import EAN13DECODER
from utils.decode import DECODEDONE
import time

import multiprocessing as mp

def solver(img_folder, line_num, angles, tilts, torelances, show_res=False, show_drop=False, use_mp=True):
    accuracy = 0
    all_codes = []
    all_images = []
    curr_time = time.time()

    for img_path in tqdm(os.listdir(img_folder)):
        all_images.append(img_path)
        img = cv2.imread(os.path.join(img_folder, img_path))
        
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.mean(img, axis=2).astype(np.uint8)

        H, W = img.shape[:2]
        if H > W:
            img = ip.resize(img, (800, 600))
        else:
            img = ip.resize(img, (600, 800))


        img = ip.normalize(img)

        img = ip.gaussian_blur(img)

        img = ip.adaptive_binarize(img, th_shift=-0.1)

        img = 1 - img

        # plt.imshow(img, cmap="gray")
        # plt.show()

        status = 'fail'
        break_all = False

        for tilt in tilts:
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
                                if show_res:
                                    show_text = 'OK ' + final_code
                                    plt.imshow(img, cmap="gray")
                                    plt.text(5, -2, show_text, ha='left', va='bottom', fontsize=14, color="g")
                                    plt.show()
                                raise DECODEDONE('done')

                        except Exception as e:
                            # if is DECODERERROR, then continue
                            if e.__class__.__name__ == "DECODERERROR":
                                # print(e)
                                continue
                            else:
                                if e.__class__.__name__ == "DECODEDONE":
                                    # print("Done!")
                                    accuracy += 1
                                    all_codes.append(final_code)
                                    break_all = True
                                    break
                                else:
                                    raise e
                                
                    if break_all:
                        break

                if break_all:
                    break

            if break_all:
                break

        if status == 'fail':
            
            if show_res:
                show_text = 'FAIL'
                plt.imshow(img, cmap="gray")
                plt.text(5, -2, show_text, ha='left', va='bottom', fontsize=14, color="r")
                plt.show()
            all_codes.append("fail")
            # print("Fail! Please drop this class.")
            if show_drop:
                import webbrowser
                webbrowser.open_new('./fail.pdf')

    avg_duration = time.time() - curr_time
    avg_duration /= len(os.listdir(img_folder))
    return accuracy/len(os.listdir(img_folder)), avg_duration, all_codes, all_images


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
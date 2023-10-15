if __name__ == "__main__":
    # initialize pyqt5 (for matplotlib)
    import sys
    from PyQt5 import QtWidgets
    QtWidgets.QApplication(sys.argv)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import function as f

class EAN13DECODER():
    def __init__(self, barcode=None, tolerance=3, flip=False):
        if barcode is not None:
            self.barcode = barcode.astype(np.int8)
            self.pxlen = len(barcode)
            self.flip = flip
            self.torelance = tolerance
            self.units = self.find_units()

        else:
            self.barcode = None
            self.pxlen = None
            self.flip = None
            self.torelance = None
            self.units = None

    def find_units(self):
        intervals, _ = self.get_intervals()
        # sns.histplot(intervals)
        # plt.show()
        # find the first 4 peaks in the histogram
        peaks = self.get_peaks(intervals)
        # print(peaks)
        if len(peaks) < 1:
            raise DECODERERROR("cannot find peaks in the histogram")

        # the highest 4 peaks
        peaks.sort(key=lambda x: x['value'], reverse=True)
        highest_peaks = peaks[:4]
        highest_peaks = [i['index'] for i in highest_peaks]

        # the foremost 4 peaks
        peaks.sort(key=lambda x: x['index'])
        foremost_peaks = peaks[:4]
        foremost_peaks = [i['index'] for i in foremost_peaks]
        
        highest_peaks.sort()
        foremost_peaks.sort()

        # print(highest_peaks)
        # print(foremost_peaks)
        smallest_unit = highest_peaks[0] + self.torelance/2
        units = [smallest_unit, smallest_unit*2, smallest_unit*3, smallest_unit*4]
        # print(units)
        return units
        

    def get_intervals(self, sort_interval=True, barcode=None):
        
        if barcode is None:
            barcode = self.barcode

        intervals = []
        try:
            content = [barcode[0]]
        except IndexError as e:
            # print(barcode)
            raise DECODERERROR("barcode is empty")

        interval_count = 1
        for i in range(1, len(barcode)):
            if barcode[i] != barcode[i-1]:
                intervals.append(interval_count)
                content.append(barcode[i])
                interval_count = 1
            else:
                interval_count += 1
        intervals.append(interval_count)
        

        
        if sort_interval:
            intervals.sort()

        return intervals, content

    def get_peaks(self, intervals):
        # intervals to histogram function
        hist = np.zeros(max(intervals) + 1)
        for i in intervals:
            hist[int(i/self.torelance)] += 1
        # plt.plot(hist)
        # plt.show()
        # find the first 4 peaks
        peaks = []
        for i in range(1, len(hist)-1):
            # i/4 * 95 should no greater than pxlen (95 is the bar number of EAN13)
            if i/4 * 95 > self.pxlen:
                # print(f'interval {i} is too large, stop searching')
                break

            if hist[i] > hist[i-1] and hist[i] >= hist[i+1]:
                peaks.append({'index': i*self.torelance, 'value': hist[i]})
        return peaks

    def barcode_locate(self, barcode=None):
        if barcode is None:
            barcode = self.barcode
        intervals, content = self.get_intervals(sort_interval=False, barcode=barcode)

        # drop the foremost and the last intervals that bigger than 5 times of the smallest unit
        intervals_new = []
        content_new = []
        intervals_tmp = []
        content_tmp = []
        is_interupt = False
        for i in range(len(intervals)):
            if intervals[i] < self.units[0]*5:
                intervals_tmp.append(intervals[i])
                content_tmp.append(content[i])
                is_interupt = True
            else:
                if is_interupt:
                    if len(intervals_tmp) > 0:
                        intervals_new.append(intervals_tmp)
                        content_new.append(content_tmp)
                    intervals_tmp = []
                    content_tmp = []
                
                is_interupt = False
        
        if len(intervals_new) <= 0:
            raise DECODERERROR("cannot find any barcode")
        
        return intervals_new, content_new

    # get_barcode using split the whole series into 95 bars
    # a little bit un-toraleant
    def get_barcode(self, barcode=None):

        if barcode is None:
            barcode = self.barcode
        
        interval, content = self.barcode_locate(barcode)
        
        for interv, cont in zip(interval, content):
            interv, cont = self.extend_0(interv, cont, th=1, ext=1.1)
            # interv, cont = self.shortent_0(interv, cont, th=1, ext=1)
            # self.plot_line_from_intev_and_cont(interv, cont)
            if np.sum(interv) < 95:
                continue
            
            # start sign
            if not (np.array(interv[0:3]) < self.units[0]+self.torelance/2).all() or not (cont[0:3] == [1,0,1]):
                # print("start sign error")
                continue

            # middle sign
            mid_position = []
            for i in range(len(interv)-5):
                if (np.array(interv[i:i+5]) < self.units[0]+self.torelance/2).all() and (cont[i:i+5] == [0,1,0,1,0]):
                    mid_position.append(i)

            if len(mid_position) <= 0:
                # print("middle sign error")
                continue

            # end sign
            if not (np.array(interv[-3:]) < self.units[0]+self.torelance/2).all() or not (cont[-3:] == [1,0,1]):
                # print("end sign error")
                continue


            # print(mid_position)

            for mid_pos in mid_position:

                proc_barcode = np.zeros(95).astype(np.int8)
                proc_barcode[0:3] = [1,0,1]
                proc_barcode[-3:] = [1,0,1]
                proc_barcode[45:50] = [0,1,0,1,0]

                interv_l = interv[3:mid_pos]
                interv_r = interv[mid_pos+5:-3]
                cont_l = cont[3:mid_pos]
                cont_r = cont[mid_pos+5:-3]

                for i, j, index in zip([interv_l, interv_r], [cont_l, cont_r], [0,1]):
                    length = np.sum(i)
                    bars = np.linspace(0, length, 43)
                    bars = np.round(bars).astype(np.int16)
                    barcode_half = self.gen_code_from_intev_and_cont(i, j)
                    proc_barcode_half = np.zeros(42).astype(np.int8)
                    loss = 0
                    for k in range(len(bars)-1):
                        val = np.mean(barcode_half[bars[k]:bars[k+1]])
                        loss += (0.5 - np.abs(val - 0.5))
                        val = np.round(val).astype(np.int8)
                        proc_barcode_half[k] = val

                    if index == 0:
                        proc_barcode[3:45] = proc_barcode_half
                    else:
                        proc_barcode[50:-3] = proc_barcode_half

                proc_barcode = np.array(proc_barcode)
                # print("".join([str(i) for i in proc_barcode]))

                # barcode_plot = np.tile(proc_barcode, (30, 1))
                # plt.imshow(barcode_plot, cmap='gray')
                # plt.vlines(bars, 0, 30, colors='r', linewidth=1)
                plt.show()
                # print(proc_barcode)
                # print(loss)
                
                # print('find barcode!')
                final_code = "".join([str(i) for i in proc_barcode])
                
                # print(final_code)
                final_code_decoded = self.decode(final_code)
                # print(final_code)

                return final_code, final_code_decoded, 'success'
        

        # all of the possible barcode are not valid
        # return self.get_barcode_2(barcode)
        raise DECODERERROR("all of the possible barcode are not valid")

    # get barcode but find start, middle and end sign first
    def get_barcode_2(self, barcode=None):

        if barcode is None:
            barcode = self.barcode
        
        interval, content = self.barcode_locate(barcode)
        
        for interv, cont in zip(interval, content):
            interv, cont = self.extend_0(interv, cont, th=0.8, ext=1.1)
            # interv, cont = self.shortent_0(interv, cont, th=0.8, ext=1)
            # self.plot_line_from_intev_and_cont(interv, cont)
            if np.sum(interv) < 95:
                continue
            
            # interv_acc = np.cumsum(interv)
            length = np.sum(interv)
            # bar_width = length / 95.
            bars = np.linspace(0, length, 96)
            bars = np.round(bars).astype(np.int16)
            barcode = self.gen_code_from_intev_and_cont(interv, cont)
            proc_barcode = np.zeros(95).astype(np.int8)
            loss = 0
            for i in range(len(bars)-1):
                val = np.mean(barcode[bars[i]:bars[i+1]])
                loss += (0.5 - np.abs(val - 0.5))
                val = np.round(val).astype(np.int8)
                proc_barcode[i] = val

            # barcode_plot = np.tile(barcode, (30, 1))
            # plt.imshow(barcode_plot, cmap='gray')
            # plt.vlines(bars, 0, 30, colors='r', linewidth=1)
            # plt.show()
            # print(proc_barcode)
            # print(loss)
            
            # start sign
            if proc_barcode[:3].tolist() != [1,0,1]:
                # print("start sign error")
                continue

            # middle sign
            if proc_barcode[45:50].tolist() == [0,1,0,1,0]:
                continue

            # end sign
            if proc_barcode[-3:].tolist() != [1,0,1]:
                # print("end sign error")
                continue
            
            print('find barcode!')
            final_code = "".join([str(i) for i in proc_barcode])
            
            # print(final_code)
            final_code_decoded = self.decode(final_code)
            print(final_code)

            return final_code, final_code_decoded, 'success'
        

        # all of the possible barcode are not valid
        raise DECODERERROR("all of the possible barcode are not valid")


    def decode(self, barcode, recursion=False):

        if self.flip:
            barcode = barcode[::-1]

        left_parity_pattern = {
        '0001101': '0o', '0011001': '1o', '0010011': '2o', '0111101': '3o', '0100011': '4o',
        '0110001': '5o', '0101111': '6o', '0111011': '7o', '0110111': '8o', '0001011': '9o',
        '0100111': '0e', '0110011': '1e', '0011011': '2e', '0100001': '3e', '0011101': '4e',
        '0111001': '5e', '0000101': '6e', '0010001': '7e', '0001001': '8e', '0010111': '9e'
        }
        
        right_parity_pattern = {
        '1110010': 0, '1100110': 1, '1101100': 2, '1000010': 3, '1011100': 4,
        '1001110': 5, '1010000': 6, '1000100': 7, '1001000': 8, '1110100': 9
        }

        first_digit_pattern = {
        'oooooo': 0, 'ooeoee': 1, 'ooeeoe': 2, 'ooeeeo': 3, 'oeooee': 4,
        'oeeooe': 5, 'oeeeoo': 6, 'oeoeoe': 7, 'oeoeeo': 8, 'oeeoeo': 9
        }
        
        left_side = barcode[3:45]
        right_side = barcode[50:-3]

        left_side = [left_side[i:i+7] for i in range(0, len(left_side), 7)]
        right_side = [right_side[i:i+7] for i in range(0, len(right_side), 7)]

        left_code = ''
        right_code = ''



        try:
            for i in left_side:
                left_code += left_parity_pattern[i]
            for i in right_side:
                right_code += str(right_parity_pattern[i])

            first_code = first_digit_pattern[left_code[1::2]]
            left_code = left_code[0::2]

            final_code = ''.join([str(first_code), str(left_code), str(right_code)])

            if self.checksum(final_code):
                return final_code
            else:
                raise DECODERERROR('ChecksumError')

        except Exception as e:
            if e.__class__.__name__ == 'KeyError' and not recursion:
                self.flip = not self.flip
                return self.decode(barcode, recursion=True)
            
            elif e.__class__.__str__ == 'ChecksumError' and not recursion:
                self.flip = not self.flip
                return self.decode(barcode, recursion=True)

            else:
                raise DECODERERROR("cannot decode barcode")


    
    def checksum(self, digits):
        # check if the barcode is valid
        # barcode: str
        # return: bool
        if len(digits) != 13:
            return False
        
        digits = [int(i) for i in digits]


        reversed_digits = digits[::-1]
        odd_sum = sum(int(reversed_digits[i]) for i in range(0, len(reversed_digits), 2)) * 3
        even_sum = sum(int(reversed_digits[i]) for i in range(1, len(reversed_digits), 2))
        total = odd_sum + even_sum
        check_digit = 10 - (total % 10)

        if check_digit == 10:
            check_digit = 0

        return check_digit


    def gen_code_from_intev_and_cont(self, intervals, content):
        barcode = []
        for i in range(len(intervals)):
                barcode += [content[i]] * intervals[i]

        return barcode

    def plot_line(self, barcode=None):
        if barcode is None:
            barcode = self.barcode

        barcode = np.tile(barcode, (30, 1))
        plt.imshow(barcode, cmap='gray')
        plt.show()

    def plot_line_from_intev_and_cont(self, intervals, content):
        barcode = self.gen_code_from_intev_and_cont(intervals, content)
        self.plot_line(barcode)

    def extend_0(self, interval, content, th=1, ext=1):
        for i in range(len(interval)):
            if (interval[i] < (self.units[0] * th)) & (content[i] == 0):
                interval[i] = int(self.units[0] * ext)

        return interval, content
    
    def shortent_0(self, interval, content, th=1, ext=1):
        for i in range(len(interval)):
            if (interval[i] > self.units[0]*4*th) & (content[i] == 0):
                interval[i] = int(self.units[0]*4 * ext)

        return interval, content

def encoder():
    # read barcodes from txt
    barcodes = []
    with open('./barcodes.txt', 'r') as f:
        for line in f:
            barcodes.append(line.strip())

    print(len(barcodes))

    # generate barcodes
    left_parity_pattern = {
        '0001101': '0o', '0011001': '1o', '0010011': '2o', '0111101': '3o', '0100011': '4o',
        '0110001': '5o', '0101111': '6o', '0111011': '7o', '0110111': '8o', '0001011': '9o',
        '0100111': '0e', '0110011': '1e', '0011011': '2e', '0100001': '3e', '0011101': '4e',
        '0111001': '5e', '0000101': '6e', '0010001': '7e', '0001001': '8e', '0010111': '9e'
        }
        
    right_parity_pattern = {
    '1110010': 0, '1100110': 1, '1101100': 2, '1000010': 3, '1011100': 4,
    '1001110': 5, '1010000': 6, '1000100': 7, '1001000': 8, '1110100': 9
    }

    first_digit_pattern = {
    'oooooo': 0, 'ooeoee': 1, 'ooeeoe': 2, 'ooeeeo': 3, 'oeooee': 4,
    'oeeooe': 5, 'oeeeoo': 6, 'oeoeoe': 7, 'oeoeeo': 8, 'oeeoeo': 9
    }
    
    for code in barcodes:
        proc_code = ''
        parity_code = code[0]
        parity_pattern = list(first_digit_pattern.keys())[list(first_digit_pattern.values()).index(int(parity_code))]
        # print(parity_pattern)
        code = code[1:]
        proc_code += '101'
        # generate left side
        for i in range(6):
            index = str(code[i]) + parity_pattern[i]
            pattern = list(left_parity_pattern.keys())[list(left_parity_pattern.values()).index(index)]
            proc_code += pattern

        proc_code += '01010'

        # generate right side
        for i in range(6, 12):
            index = code[i]
            pattern = list(right_parity_pattern.keys())[list(right_parity_pattern.values()).index(int(index))]
            proc_code += pattern

        proc_code += '101'

        print(proc_code)



class DECODERERROR(Exception):
    def __init__(self, message='decoder error'):
        self.message = message

    def __str__(self):
        return self.message
    
class DECODEDONE(Exception):
    def __init__(self, message='done!'):
        self.message = message

    def __str__(self):
        return self.message

if __name__ == "__main__":
    encoder()
    # test = np.load('lines.npy', allow_pickle=True)[4]
    # test = np.array(test)
    # decoder = EAN13DECODER(test, tolerance=5)
    # final_code = decoder.get_barcode()
    # print(final_code)
    # print(len(final_code))
    # decoder.plot_line()
    # print(test)
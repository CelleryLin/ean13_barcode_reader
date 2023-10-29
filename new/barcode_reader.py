if __name__ == "__main__":
    # initialize pyqt5 (for matplotlib)
    import sys
    from PyQt5 import QtWidgets
    QtWidgets.QApplication(sys.argv)

from utils.slover import solver
import sys
import time

if __name__ == "__main__":

    args = sys.argv
    line_num = None
    angles = None
    tilts = None
    torelances = None
    path = None

    if len(args) >= 1:
        for arg in args:
            if arg in ('-h','--help'):
                pass
            elif arg in ('-l','--line'):
                line_num = int(args[args.index(arg)+1])
            elif arg in ('-a','--angles'):
                angles = args[args.index(arg)+1]
            elif arg in ('-t','--tilts'):
                tilts = args[args.index(arg)+1]
            elif arg in ('-tol','--torelances'):
                torelances = args[args.index(arg)+1]
            elif arg in ('-p','--path'):
                path = args[args.index(arg)+1]
            else:
                pass

    line_num = 100 if line_num is None else line_num
    # 0<=angle<=180
    angles = [ 90, 60, 120, 30, 150, 0, 180] if angles is None else [int(i) for i in angles.split(',')]
    tilts = [None] if tilts is None else [int(i) for i in tilts.split(',')]
    torelances = [2, 3, 4, 5, 1] if torelances is None else [int(i) for i in torelances.split(',')]
    

    print('line_num: ', line_num)
    print('angles: ', angles)
    print('tilts: ', tilts)
    print('torelances: ', torelances)
    print('path: ', path)

    print('\nstaring solving...')
    acc,duration,_ = solver(path, line_num=line_num, angles=angles, tilts=tilts, torelances=torelances)
    print('\ndone!')
    print('accuracy: ', acc)
    print('avg_duration: ', duration)
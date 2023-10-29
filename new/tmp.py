folder = './barcodes_more'

# random delete files in folder

import random
import os

all_files = os.listdir(folder)
all_files.remove('bypass')
all_files_sampled = random.sample(all_files, 100)
all_flies_remove = list(set(all_files) - set(all_files_sampled))

# leave only 100 files
for i in range(len(all_flies_remove)):
    os.remove(os.path.join(folder, all_flies_remove[i]))
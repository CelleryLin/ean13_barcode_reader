import numpy as np

class labelTable():
    def __init__(self, table):
        self.table = table

        # find the number of objects
        tmp = []
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                if table[i,j] is not None:
                    if table[i,j].get_label() not in tmp:
                        tmp.append(table[i,j].get_label())
        
        self.obj_num = len(tmp)
        self.obj_list = tmp
        self.area = []
        self.location = []
        self.shape = table.shape

        for n in self.obj_list:
            area = 0
            bar_x = 0
            bar_y = 0
            for i in range(table.shape[0]):
                for j in range(table.shape[1]):
                    if (table[i,j] is not None) and (table[i,j].get_label() == n):
                        area += 1
                        bar_x += j
                        bar_y += i
            
            self.area.append(area)
            self.location.append((bar_x/area, bar_y/area))

    def __getitem__(self, inxed):
        return self.table[inxed]
    
    def get_table(self):
        return self.table
    
    def get_area(self):
        return self.area

    def get_location(self):
        return self.location

    def get_int_array(self):
        int_array = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.table[i,j] is not None:
                    int_array[i,j] = self.table[i,j].get_label()
        return int_array
    
    def get_obj_mask(self, n):
        mask = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.table[i,j] is not None:
                    if self.table[i,j].get_label() == self.obj_list[n]:
                        mask[i,j] = 1
        return mask
    

class labelNode(labelTable):
    def __init__(self, label):
        self.label = label
        self.prev = None
        self.next = None

    def set_prev(self, prev):
        self.prev = prev

    def set_next(self, next):
        self.next = next

    def insert(self, n):
        if self.has_same_root(n):
            pass
        else:
            tmp = self.next
            root_n = n.get_root()
            leaf_n = n.get_leaf()
            self.set_next(root_n)
            leaf_n.set_next(tmp)
            root_n.set_prev(self)

    def get_label(self):
        return self.label
    
    def get_root(self):
        if self.prev == None:
            return self
        else:
            return self.prev.get_root()
        
    def get_leaf(self):
        if self.next == None:
            return self
        else:
            return self.next.get_leaf()
    
    def has_same_root(self, n):
        if self.get_root().get_label() == n.get_root().get_label():
            return True
        else:
            return False

    def has_label(self, label):
        if self.label != 0:
            return True
        else:
            return False
        
    # def loop_detect(self, label, min_label):
    #     if self.prev == None:
    #         return False, min_label
    #     elif self.prev.get_label() == label:
    #         return True, min_label
    #     else:
    #         if self.prev.get_label() < min_label.get_label():
    #             min_label = self.prev
            
    #         return self.prev.loop_detect(label, min_label)
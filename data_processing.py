import numpy as np
import pygtrie as trie

class data_base(object):
    def __init__(self, id_index):
        self.id_index = id_index
        self.data = self.read_files()
        self.t = trie.CharTrie()
        self.count = 0
        unique_data = []
        for x in self.data:
            if self.t.has_key(x)==False:
                self.t[x] = self.count
                unique_data.append(x)
                self.count = self.count+1
        self.unique_data = np.array(unique_data)


    def read_files(self):
        # read raw data
        behavior_dataset = np.array([0,0,0])
        for driver_id in (self.id_index):
            filename = 'output2/driver_%d_map.txt' % driver_id
            raw_file = np.loadtxt(filename, delimiter=',')
            behavior_dataset = np.vstack((behavior_dataset, raw_file))
        return behavior_dataset

    def ind2color(self, index):
        return self.unique_data[index,:]

    def color2ind(self,color):
        return self.t.get(color)

    def data_size(self):
        return np.shape(self.data)[0]

    def random_data(self,size):
        data =  np.array(self.ind2color(np.random.randint(self.count,size=(size,1))))
        return data.tolist()

    def is_valid(self,color):
        return self.t.has_key(color)



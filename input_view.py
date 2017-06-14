import numpy as np
from update_cmap import DataBase

for id in range(10):
    obj2 = DataBase(data=np.loadtxt("output2/driver_%d_map.txt" % id, delimiter=","), driver_id=id)
    obj2.show()
obj2.view()


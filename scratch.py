import numpy as np

from map import Map

buf = []

buf.append([1,1,2,2,3])
buf.append([4,7,8,11,3])
buf.append([4,6,2,2,3])
buf.append([3,6,8,5,3])
buf.append([4,6,2,5,3])
print buf
print(np.array(buf)[:,2])

x = 5
print x
d = Map({'y':x})
print d
print d['y']

import numpy as np, sys
L = open(sys.argv[1]).readlines(); n = int(L[1])
xyz = np.array([[float(l[20+8*i:28+8*i]) for i in range(3)] for l in L[2:2+n]])
box = np.array([float(v) for v in L[2+n].split()[:3]])
print('box', np.round(box,2))
print('min', np.round(xyz.min(0),2))
print('max', np.round(xyz.max(0),2))

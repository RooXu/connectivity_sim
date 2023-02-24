import torch 
import matplotlib.pyplot as plt
import numpy as np
from random import *

def makeDistanceKernel(size,dmax = 1000): # Make Distance Kernel 
    # size: the width and height of the desired kernel 
    # dmax: the maximum desired distance 

    if size % 2 == 0: # Forces the kernel to have be odd shaped. 
        size += 1 
    
    centerindex = int(size/2) 
    prekernelX = torch.empty((size,size),dtype=torch.float)
    prekernelY = torch.empty((size,size),dtype=torch.float)
    for i in range(size):
        prekernelY[i,:] = (float(centerindex-i)) ** 2 
    
    prekernelX=prekernelY.T
    kernel = (prekernelX  + prekernelY ) ** (1/2)
    print(kernel.shape)
    kernel.clamp_(max=dmax)
    return kernel

#kernSize = 351
#worldSize = 7000
#kernel = makeDistanceKernel(kernSize)
#print(kernel)
#plt.imshow(kernel)
#plt.show()
#zeros = torch.full((1,1,worldSize,worldSize),kernel[0,0])
#weights = kernel.view(1,1,5,5).repeat(1,1,1,1)
#weights = kernel.view(1,1,kernSize,kernSize).repeat(1,1,1,1)
#location = [randint(0,worldSize),randint(0,worldSize)]

#location = [900,1890]
#print(location)
#zeros[0,0,location[0],location[1]] = 1
#zeros.to_sparse()
#print(zeros)

#convoved = torch.nn.functional.conv2d(zeros,weights)
#plt.imshow(convoved[0,0])
#plt.show()

def makePasteBounds(location, kernSize, worldSize:tuple, kernel:torch.tensor):
    rowinit = location[0]-int(kernSize/2)
    rowfin = location[0]+int(kernSize/2)+1
    colinit = location[1]-int(kernSize/2)
    colfin = location[1]+int(kernSize/2)+1

    if rowinit < 0:
        kernel = kernel[abs(rowinit):,:]
        rowinit = 0 

    if colinit < 0:
        kernel = kernel[:,abs(colinit):]
        colinit=0

    if rowfin >= worldSize[0]:
        kernel = kernel[0:rowfin-worldSize[0],:]
        rowfin = worldSize[0]-1

    if colfin >= worldSize[1]:
        kernel = kernel[:,0:colfin-worldSize[1]]
        colfin = worldSize[1]-1

    return [rowinit,rowfin,colinit,colfin,kernel]

#print(kernel.cpu)
#pastbounds = makePasteBounds(location,kernSize,(worldSize,worldSize),kernel)
#print(pastbounds[0:4],print(pastbounds[4].cpu))
#zeros[0,0,(pastbounds[0]):(pastbounds[1]),(pastbounds[2]):(pastbounds[3])] = pastbounds[4]

#plt.imshow(zeros[0,0]) 
#plt.show()


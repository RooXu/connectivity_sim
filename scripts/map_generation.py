#####
# Generates Maps used for my senior thesis 
##### 
from perlin_noise import PerlinNoise
import torch 
from numpy import empty,min,ptp,array,random,zeros
#from repast4py import value_layer as valyr
#from mpi4py import MPI
import matplotlib.pyplot as plt


def makePerlinRaster(width :int ,height : int , seed: int) -> torch.FloatTensor:

    noise1 = PerlinNoise(octaves=3, seed=seed)
    noise2 = PerlinNoise(octaves=6, seed=seed)
    noise3 = PerlinNoise(octaves=12, seed=seed)
    noise4 = PerlinNoise(octaves=24, seed=seed)
    #print(noise.seed)
    xpix, ypix = width, height
    pic = empty((height,width)) # possible width height mismatch 
    for i in range(ypix):
        for j in range(xpix):
            noise_val = 1.*noise1([i/xpix,j/ypix])
            noise_val += 1.0*noise2([i/xpix,j/ypix])
            noise_val += 1.0*noise3([i/xpix,j/ypix])
            pic[i,j] = noise_val
    #np.savetxt("../output/perlinRasterSeed_"+str(noise.seed)+".csv",pic,delimiter = ',')
    # normalize to [0,1] 
    pic = (pic - min(pic))/ptp(pic)
    return torch.FloatTensor(pic)

def get_density_from_file() -> torch.tensor:
    # to be added
    return pic

def get_regions_from_file() -> torch.tensor: #[Region type 1, Region type 2, ... Region type N]
    # to be added
    return regions 

def make_region_map(regionBoundaries:tuple, width,height) -> torch.tensor:
    print(regionBoundaries[0,0])
    region = torch.zeros((len(regionBoundaries),width,height))
    print(region)
    for idx, regionBoundary in enumerate(regionBoundaries):
        print(regionBoundary)
        region[idx,regionBoundary[0]:regionBoundary[1]+1,regionBoundary[2]: regionBoundary[3]+1] = True
    return region

def select_random_pts_region(atDensity,regionMaps):
    # create (Nregions,row,col) tensor to store results
    matWithPoints = torch.zeros_like(regionMaps)

    # for each region type
    for mapidx, A in enumerate(regionMaps): 
        print(A)
        #start coordinate index at 0
        coordinates_idx = 0
        # declare a numpy array to store coordinates
        print(int((torch.sum(A))))
        coordinates = zeros((int((torch.sum(A))),3)).astype(int)

        # for each row in a region type map
        for rowidx, row in enumerate(A):
            # for each value in the row in a region type map
            for colidx, val in enumerate(row):
                # if the value is true, store coordinate in the coordinate array
                #print(val)
                #print(val==1.)
                if val == 1.: 
                    coordinates[coordinates_idx,:] = [coordinates_idx,rowidx,colidx]
                    coordinates_idx +=1
        # randomly choose Density*(Region Area) times, no replacement
        #print(coordinates)
        #print(int(len(coordinates[:,0]) * atDensity))
        chosen_indexes = random.choice(coordinates[:,0],size=int(len(coordinates[:,0]) * atDensity),replace=False)
        #print("chosen")
        #print(chosen_indexes)
        # set the chosen points to 1 in matWithPoints
        for idx in chosen_indexes:
            matWithPoints[mapidx, coordinates[idx,1], coordinates[idx,2]] = 1.0

    return matWithPoints

def convolve():
    return convolvedMat



""" class raster(valyr.ReadWriteValueLayer):

    def __init__(self, comm : MPI.Intracomm, bounds, borders, buffer_size, seed):
        self.width = bounds
        self.height = bounds
        self.seed= seed

        # Use perlin noise to generate an heterogeneous environment. Variables are unitless varying from 0 to 1
        density = self.makePerlinRaster(bounds, self.seed)

        super.__init__(self, comm, bounds, borders, buffer_size, initialValues)

 """
#test = makePerlinRaster(500,500,7) 

#plt.imshow(test)
#plt.colorbar()
#plt.show()

width = 100
height = 100

# 1. Create two N by N perlin grids for food density and predation risk, respectivley 
enviroVals = makePerlinRaster(width,height,seed=2)
# 2. Designate Neighborhood region 
regionBoundaries = array([[0, 99, 33, 66]])
regions = make_region_map(regionBoundaries,width,height)
# 3. Make a list of random points sampled from the region. 
regionWithDiracDeltas = select_random_pts_region(0.25,regions) 

plt.imshow(regions[0])
plt.show()
plt.imshow(regionWithDiracDeltas[0])
plt.show()
# 4. Convolute over those points using a kernel to create a mask --- i.e. in binary 
#regionWithFeatures = convolve(regionWithDiracDeltas)

# 5. Modify mask for nutrion 
binaryMask = 1-regionWithDiracDeltas
finalFoodDensity = binaryMask[0] * enviroVals
plt.imshow(finalFoodDensity)
plt.show()
# 6. modify mask for predation risk. new_mask = (1- mask)
finalPredationRisk = (1-binaryMask) * predationRisk


##########
# Foraging Model 0  
# Aaron Xu 
# 
# This foraging model samples a gaussian distribution for turning and a travel distance. 
##########

from typing import Dict, Tuple
from mpi4py import MPI
import numpy as np
from dataclasses import dataclass

from repast4py import core, random, space, schedule, logging, parameters
from repast4py import context as ctx
import repast4py
from repast4py.space import DiscretePoint as dpt
import torch 

from typing import Optional

from perlin_noise import PerlinNoise

import mk_insert_distance
import map_generation

import matplotlib.pyplot as plt
# No need for agreggate data yet
"""@dataclass
class travelLog:
    xLoc: int = 0
    yLoc: int = 0"""

DT = 2
GAMA = 1
ALPHA = 0.1
BETA = 4.0

DMAX = -round(DT * np.log(0.01) / GAMA)
DMAX = 100
EXPECTATION = 0.01
KERNEL = mk_insert_distance.makeDistanceKernel(int(DMAX),DMAX)

class Walker(core.Agent): 
    TYPE = 0 
    OFFSETS = np.array([1, -1])
    
    
    #distribution = None

    # The initialization method that runs upon object initialization
    # The arguments are what I want for my custom walker.
    def __init__(self, local_object_id: int, rank: int):#, buffered_grid_shape, initMemory = None): 
        # Initialize the parent class 
        
        super().__init__(id=local_object_id,type=Walker.TYPE,rank=rank)
        #self.pt #Init the variable used for storing agent location.
        #self.gridShape = [grid.get_local_bounds().yextent, grid.get_local_bounds().xextent]
        #self.gridShape = buffered_grid_shape

        print("#####################")
        #print(self.gridShape)
        #print(self.gridShape)
        '''if initMemory == None:
            self.qualityMem =  torch.zeros(self.gridShape, dtype=torch.float) # initialize variable used for storing agent memory. Starts at 0
        else:
            self.qualityMem = initMemory'''
        self.qualityMem = None
        #self.attractionMem =  torch.zeros(gridShape, dtype=torch.half) # initialize variable used for storing agent memory. Starts at 0
            # may not need to initialize this 
        self.alpha = ALPHA
        self.beta  = BETA
        self.gama = GAMA
        self.dT = DT
        self.expectation = EXPECTATION

    def save(self) -> Tuple:
        """Saves the state of this Walker as a Tuple.
        Returns:
            The saved state of this Walker.
        """
        return (self.uid, self.qualityMem)
    

    def rndwalk(self, grid):
        # sample from a distribution for theta and r 
        # convert to x and y coordinates 
        # move in those directions 

        # <WIP Code> 
        #xy_dirs = random.default_rng.choice(Walker.OFFSETS, size = 2)

        np.random.seed()
        x_dirs = round(np.random.normal(0.0, 2.0))
        y_dirs = round(np.random.normal(0.0, 2.0))

        self.pt = grid.move(self, dpt(grid.get_location(self).x + x_dirs,\
                                      grid.get_location(self).y + y_dirs,\
                                      0))
        
    def memoryWalk(self, worldgrid, qualityvalues): 
        gridShape = qualityvalues.grid.shape
        if self.qualityMem == None:
            self.qualityMem = torch.zeros(gridShape, dtype=torch.float)
        
        print(["Walker ID", self.uid])
        # Import qualityValuesFromGrid range[0,1] or bool
        quality = qualityvalues.grid.clone()
        
        #print("percep shape")
        #print(quality.shape)
        #plt.imshow(self.qualityMem)
        #plt.title("quality Mem")
        #plt.show()
        #print("Step 1: Import Quality Grid")
        #print(quality)
        #print("memoryWalk: the grid shape is: ", self.gridShape)
        # calculate distnace matrix
        perceptionMat = mk_insert_distance.pasteKernel([worldgrid.get_location(self).x % gridShape[0],\
                                                        worldgrid.get_location(self).y% gridShape[1]], \
                                                        len(KERNEL),\
                                                        gridShape, \
                                                        KERNEL)
        #plt.imshow(perceptionMat)
        
        #print("Step 2: The Distance matrix d_i,j is:")
       
        #print(perceptionMat.shape)
        #print(quality.shape)
        #mulitply true quality with the perceptionTensor 

        distanceDecay = torch.exp(-self.alpha*perceptionMat)*(1/self.dT)
        quality = quality * distanceDecay
        
        
        #print("Step 3: Multiply quality by perceptionMatrix")
        #print(quality) 
        decayCoeff =  np.exp(-self.beta*self.dT) #scalar

        quashedExpec = (1-decayCoeff)*self.expectation #scalar 
        
        quality = quality + (1 - (-1) * distanceDecay) * (decayCoeff*self.qualityMem + (quashedExpec))
        #plt.imshow(quality)
        #plt.show()
        
        # quality + [(-1)*perceptionMat + 1 ]*[previousqualityMem*forgetting + expectation_envelope*expectation]    
        #print("Step 4: Quality + Decay Term:")
        #print(quality)

        self.qualityMem = quality.clone()

        costFunc = torch.exp(-self.gama * perceptionMat / self.dT)# is a 2D tensor
        #plt.imshow(costFunc)
       
        #print("the cost function")
        #print(costFunc)
        attraction = quality * costFunc #now is the attraction matrix. currently only supports one layer 
        attraction = torch.clip(attraction,min=0,max=1)
        attraction[attraction < 0.001] = 0
        #plt.imshow(attraction)
        #plt.title("Attraction clamped")
        #plt.show()
        #print(torch.sum(costFunc))
        #print("Step 5: Multiply by Cost Function")
        #print(quality)
        #print(torch.sum(quality))
        #print("Step 6: Make Probabilities")
        if torch.sum(attraction) == 0.: 
            print("Sum Quality is Zero. BAD")
            self.rndwalk(worldgrid)
        else:
            probability = torch.divide(attraction,torch.sum(attraction))
            
            #plt.imshow(probability)
            #plt.title("prob")
            #plt.show()
            #print(probability)
            probFlat =  probability.flatten().numpy()
            #print(np.sum(probFlat))
            #print(probFlat)
            #if the 1D index starts with zero, the algorithm to recover its unflattened index is 
            # col = idx%length(col) 
            # row = int(idx/length(col))
            idx1D = np.random.choice(a = len(probFlat), size = 1, replace = True, p = probFlat)  #<- choose which index to move to based on the probability kernel 
            dim1Size = quality.size(dim=1) 
            if dim1Size != gridShape[1]: 
                print("dim1Size != gridWidth") 
            local_newY = idx1D[0] % dim1Size
            #print("newX is:", newX)
            local_newX = int(idx1D[0]/ dim1Size)
            #print("newY is:", newY)
            print(["Dimsize", dim1Size])
            #change_newX = local_newX-self.pt.x
            #change_newY = local_newY-self.pt.y

           # print(["X",self.pt.x ,"Y",self.pt.y])
            print(["Local_NewX",local_newX,"Loc NY", local_newY])
            '''if np.sqrt(change_newX**2+change_newY**2) > DMAX+5: 
                print("Wow thats a large jump")
                
                #exit()'''
            self.pt = worldgrid.move(self, dpt(local_newX,
                                               local_newY,0))

walker_cache = {} 
# this cache maintains all pointers of the created walkers in this model! (pointers don't exist in python, but I think it works like that.)
# It seems like the cache is never accessed by any other method other and restore walker. This is, in other words, should be like a private variable. 
# To edit or update walker states, see the Model Class

def restore_walker(walker_data: Tuple): 
    """
    Args:
        walker_data: tuple containing the data returned by Walker.save.
    """
    # uid is a 3 element tuple: 0 is id, 1 is type, 2 is rank
    # Transfering data or "unzipping" data from the tuple into individual variables
    #print(walker_data)

    uid = walker_data[0]
    #print("UID")
    #print(uid)
    #pt_array = walker_data[1]
    #print("pt_array")
    #print(pt_array)
    #pt = dpt(pt_array[0], pt_array[1], 0)
   # print("memory")
    #print(memory)
    #print("gridshape")
    #print(gridShape)
    if uid in walker_cache:                     # If the uid exists in the cache
        print("@restore_walker - Restoring to Hold Skin/husk")
        walker = walker_cache[uid]              # point to an existing walker 
    else:
        print("@restore_walker - Create a new husk")
        walker = Walker(uid[0], uid[2])     # Else create a new
        walker_cache[uid] = walker
    #walker.pt = pt
    return walker

#class World(repast4py.value_layer.ReadWriteValyeLayer):
#    def __init__(self,comm: MPI.Intracomm, params: Dict, borders: repast4py.space.BorderType, buffer_size: int, init_value: float):

def initialize_grid_values(width,height,seed,density = 0.96): 
    
    baseFoodVals = map_generation.makePerlinRaster(width,height,seed)
    print("DONE : baseFoodVals")
    baseRiskVals = map_generation.makePerlinRaster(width,height,seed*2)
    print("DONE : baseRiskVals")
    regionBoundaries = np.array([[0, height-1, round(width*(1/3)), round(width*(2/3))]])
    #print("beeb")
    #print(regionBoundaries)
    #print("boop")
    regionMasks = map_generation.make_region_map(regionBoundaries,width,height)
    print("DONE : region masks")
    regionWithDiracDeltas = map_generation.select_random_pts_region(density,regionMasks)

    maskedFoodVals = baseFoodVals*(1-regionWithDiracDeltas[0])
    print("DONE : maskedFoodVals")
    maskedRiskVals = torch.clip(baseRiskVals+(1.0 * regionWithDiracDeltas[0]),min=0,max=1)
    print("DONE : maskedRiskVals")

    """plt.figure(1)
    plt.imshow(maskedFoodVals)
    plt.figure(2)
    plt.imshow(maskedRiskVals)
    plt.show()"""
    return [maskedFoodVals, maskedRiskVals]

class Model: # inherits nothing
    """
    The Model class encapsulates the simulation, and is
    responsible for initialization (scheduling events, creating agents,
    and the grid the agents inhabit), and the overall iterating
    behavior of the model.

    Args:
        comm: the mpi communicator over which the model is distributed.
        params: the simulation input parameters
    """
    def __init__(self, comm: MPI.Intracomm, params: Dict,rasters): 
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n Initializing Model ... \n")
        # create the schedule 
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1,1,self.step)
        self.runner.schedule_repeating_event(1.1,1,self.log_agents)
        # self.runner.schedule_repeating_event(<when to start>,<how often>,self.<method>)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        # create the context to hold the agents and manage cross process synchronization
        self.context = ctx.SharedContext(comm) 

        # create a bounding box equal to the size of the entire global world grid
        box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
        
        print(["@Init - ",box])
        # create a SharedGrid of 'box' size with sticky borders that allows multiple agents
        # in each grid location.
        self.grid = space.SharedGrid(name='grid', bounds=box, borders=space.BorderType.Sticky,
                                     occupancy=space.OccupancyType.Multiple, buffer_size=0, comm=comm) # buffersize should be DMAX
        
        self.context.add_projection(self.grid)    
          
        self.rank = comm.Get_rank()
        self.nproc = comm.Get_size()                     
        print(["@Init - Rank", self.rank,", Num Ranks: ", self.nproc])
    
        #print(self.raster)

        # Get Local Bounds to divie up the raster values into chunks for parallel processing. 
        localBounds = self.grid.get_local_bounds()
        print(["@Init - Local Bounds:", localBounds])
        
        #plt.imshow(rasters[0])
        #plt.show()
        onRankFood = rasters[0]#[localBounds.ymin:(localBounds.ymin+localBounds.yextent),
                                #localBounds.xmin:(localBounds.xmin+localBounds.xextent)]
        #print(onRankFood.shape)
        #plt.imshow(onRankFood)
        #plt.show()
        onRankRisk = rasters[1]#[localBounds.ymin:(localBounds.ymin+localBounds.yextent),
                               # localBounds.xmin:(localBounds.xmin+localBounds.xextent)]
        self.foodValue = repast4py.value_layer.SharedValueLayer(comm, box, borders=space.BorderType.Sticky, buffer_size = DMAX
         ,init_value = onRankFood) # Buffersize here should depend on alpha, not DMAX
        
        #print(["Shape of Food Value",self.foodValue.grid.shape])
        self.riskValue = repast4py.value_layer.SharedValueLayer(comm, box, borders=space.BorderType.Sticky, buffer_size = DMAX
         ,init_value = onRankRisk)
            # init_values should be made with image 

        #print(self.foodValue.grid.shape)
        
        # Populate world with walkers
        rng = np.random.default_rng(seed=3)
        for i in range(params['walker.count']):
            # get a random x,y location in the grid
            pt = self.grid.get_random_local_pt(rng)
            #pt = dpt(50,0)
            # create and add the walker to the context
            #walker = Walker(np.random.choice([0,1,2,3,4,5,6,7]), self.rank, pt, buffered_grid_shape=self.foodValue.grid.shape)
            walker = Walker(i, self.rank)#, buffered_grid_shape=self.foodValue.grid.shape)
            self.context.add(walker)
            self.grid.move(walker, pt)
        
        # initialize individual logging
        self.agent_logger = logging.TabularLogger(comm, params['agent_log_file'],\
            ['tick', 'agent_id', 'agent_uid_rank', 'x', 'y'])
        
        # Count initial data at time t = 0 and log
        
    def step(self):
        for walker in self.context.agents(): 
            walker.memoryWalk(self.grid,self.foodValue)
        
        self.context.synchronize(restore_walker)
        print(["@step - ", "Rank", self.rank, "Num Agents",self.context.size()])
        #self.worldMatrix.swap_layers()
        # <WIP> Insert aggregate logging stuff. 
            #tick = self.runner.schedule.tick 
            #self.data_set.log(tick)
        # Clear temporary agregate variables for next tick

    def log_agents(self): 
        tick = self.runner.schedule.tick
        print("TICK TICK TICK" ,tick)
        #if tick == 2.1: 
            #quit()
        for walker in self.context.agents():
            self.agent_logger.log_row(tick, walker.id,walker.uid_rank, self.grid.get_location(walker).x,self.grid.get_location(walker).y)
        self.agent_logger.write() #not necessary to call every time. Potential for optimization by deciding how often to write
    
    def at_end(self):
        #self.data_set.close() #commented out because no aggregate data collection yet
        for walker in self.context.agents():
            plt.imshow(walker.qualityMem)
            plt.show()
        self.agent_logger.close()
    
    def start(self):
        self.runner.execute() 

def run(params: Dict):
    # Create Vanilla ValueLayer 
    print("Initializing Rasters ...")
    rasters = initialize_grid_values(params['world.width'], params['world.height'],params['random.seed']) # can source the raster from other places     
    model = Model(MPI.COMM_WORLD, params,rasters)
    model.start()

    print(walker_cache)

if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)

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

import matplotlib.pyplot as plt
# No need for agreggate data yet
"""@dataclass
class travelLog:
    xLoc: int = 0
    yLoc: int = 0"""

DT = 0.7
GAMA = 1
ALPHA = 1
BETA = 1

DMAX = -DT * np.log(0.01) / GAMA
EXPECTATION = 0
KERNEL = mk_insert_distance.makeDistanceKernel(int(DMAX),DMAX)

class Walker(core.Agent): 
    TYPE = 0 
    OFFSETS = np.array([1, -1])
    
    
    #distribution = None

    # The initialization method that runs upon object initialization
    # The arguments are what I want for my custom walker.
    def __init__(self, local_object_id: int, rank: int, pt: dpt, grid, initMemory = None): 
        # Initialize the parent class 
        
        super().__init__(id=local_object_id,type=Walker.TYPE,rank=rank)
        self.pt = pt #Init the variable used for storing agent location.
        self.gridShape = [grid.get_local_bounds().yextent, grid.get_local_bounds().xextent]
        if initMemory == None:
            self.qualityMem =  torch.zeros(self.gridShape, dtype=torch.half) # initialize variable used for storing agent memory. Starts at 0
        else:
            self.qualityMem = initMemory
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
        return (self.uid, self.pt.coordinates, self.qualityMem, self.gridShape)
    

    def rndwalk(self, grid):
        # sample from a distribution for theta and r 
        # convert to x and y coordinates 
        # move in those directions 

        # <WIP Code> 
        #xy_dirs = random.default_rng.choice(Walker.OFFSETS, size = 2)

        np.random.seed()
        x_dirs = round(np.random.normal(0.0, 2.0))
        y_dirs = round(np.random.normal(0.0, 2.0))

        self.pt = grid.move(self, dpt(self.pt.x + x_dirs,\
                                      self.pt.y + y_dirs,\
                                      0))
        
    def memoryWalk(self, worldgrid, qualityvalues): 
        
        # Import qualityValuesFromGrid range[0,1] or bool
        quality = qualityvalues.grid
        print("Step 1: Import Quality Grid")
        print(quality)
        print("memoryWalk: the grid shape is: ", self.gridShape)
        # calculate distnace matrix
    
        perceptionMat = mk_insert_distance.pasteKernel([self.pt.x, self.pt.y], len(KERNEL), self.gridShape, KERNEL)
        print("Step 2: The Distance matrix d_i,j is:")
        print(perceptionMat)
        print(perceptionMat.shape)
        #print(quality.shape)
        #mulitply true quality with the perceptionTensor 
        quality.mul_(perceptionMat[4])
        print("Step 3: Multiply quality by perceptionMatrix")
        print(quality) 

        quashedExpec = (1 - np.exp(-self.beta*self.dT))*self.expectation #scalar 
        
        quality.add_(perceptionMat.mul(-1).add_(1).mul_(self.qualityMem.mul_(self.expectation).add_(quashedExpec))) 
        # quality + [(-1)*perceptionMat + 1 ]*[previousqualityMem*forgetting + expectation_envelope*expectation]    

        self.qualityMem = quality.clone()

        costFunc = torch.exp_(-self.gama * perceptionMat / self.dT)# is a 2D tensor
        
        quality = quality.mul(costFunc) #now is the attraction matrix. currently only supports one layer 
        print(quality)
        probability = torch.divide(quality,torch.sum(quality)) 
        print(probability)
        probFlat =  probability.flatten().numpy()
        print(probFlat)
        #if the 1D index starts with zero, the algorithm to recover its unflattened index is 
        # col = idx%length(col) 
        # row = int(idx/length(col))

        idx1D = np.random.choice(a = len(probFlat), size = 1, replace = True, p = probFlat)  #<- choose which index to move to based on the probability kernel 

        dim1Size = quality.size(dim=1) 
        if dim1Size != self.gridShape[1]: 
            print("dim1Size != gridWidth") 
        newX = idx1D[0] % dim1Size
        print("newX is:", newX)
        newY = int(idx1D[0]/ dim1Size)
        print("newY is:", newY)
        self.pt = worldgrid.move(self, dpt(newX,newY,0))

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
    uid = walker_data[0]
    pt_array = walker_data[2]
    pt = dpt(pt_array[0], pt_array[1], 0)
    memory = walker_data[3]
    gridShape = walker_data[4]
    if uid in walker_cache:                     # If the uid exists in the cache
        walker = walker_cache[uid]              # point to an existing walker 
    else:
        walker = Walker(uid[0], uid[2], pt, memory, gridShape)     # Else create a new
        walker_cache[uid] = walker

    #walker.pt = pt
    return walker

#class World(repast4py.value_layer.ReadWriteValyeLayer):
#    def __init__(self,comm: MPI.Intracomm, params: Dict, borders: repast4py.space.BorderType, buffer_size: int, init_value: float):

def makePerlinRaster(widtha,widthb,heighta,heightb): 
    noise = PerlinNoise(octaves=1, seed=8)
    xpix, ypix = widthb, heightb
    pic = [[noise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)]
    plt.imshow(pic)
    plt.show()
    return torch.FloatTensor(pic).clamp_min(0)


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
    def __init__(self, comm: MPI.Intracomm, params: Dict): 
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
        
        # create a SharedGrid of 'box' size with sticky borders that allows multiple agents
        # in each grid location.
        self.grid = space.SharedGrid(name='grid', bounds=box, borders=space.BorderType.Sticky,
                                     occupancy=space.OccupancyType.Multiple, buffer_size=2, comm=comm)
        
        self.context.add_projection(self.grid)      
        rank = comm.Get_rank()                     

        # Create Vanilla ValueLayer 
        self.raster = makePerlinRaster(0, params['world.width'], 0, params['world.height']) # can source the raster from other places 

        self.worldValuaA = repast4py.value_layer.SharedValueLayer(comm, box, borders=space.BorderType.Sticky, buffer_size = int(DMAX)
         ,init_value = self.raster)
            # init_values should be made with image 

        # Populate world with walkers
        rng = np.random.default_rng(seed=1)
        for i in range(params['walker.count']):
            # get a random x,y location in the grid
            #pt = self.grid.get_random_local_pt(rng)
            pt = dpt(7,20)
            # create and add the walker to the context
            walker = Walker(i, rank, pt, grid=self.grid)
            self.context.add(walker)
            self.grid.move(walker, pt)
        
        # initialize individual logging
        self.agent_logger = logging.TabularLogger(comm, params['agent_log_file'],\
            ['tick', 'agent_id', 'agent_uid_rank', 'x', 'y'])
        
        # Count initial data at time t = 0 and log
        
    def step(self):
        for walker in self.context.agents(): 
            walker.memoryWalk(self.grid,self.worldValuaA)
        
        self.context.synchronize(restore_walker)
        #self.worldMatrix.swap_layers()
        # <WIP> Insert aggregate logging stuff. 
            #tick = self.runner.schedule.tick 
            #self.data_set.log(tick)
        # Clear temporary agregate variables for next tick

    def log_agents(self): 
        tick = self.runner.schedule.tick
        print("TICK TICK TICK" ,tick)
        #if tick == 2.1: 
        #    quit()
        for walker in self.context.agents():
            self.agent_logger.log_row(tick, walker.id,walker.uid_rank, walker.pt.x,walker.pt.y)
        self.agent_logger.write() #not necessary to call every time. Potential for optimization by deciding how often to write
    
    def at_end(self):
        #self.data_set.close() #commented out because no aggregate data collection yet
        self.agent_logger.close()
    
    def start(self):
        self.runner.execute() 

def run(params: Dict):
    model = Model(MPI.COMM_WORLD, params)
    model.start()

if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)

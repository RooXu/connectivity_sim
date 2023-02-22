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

# No need for agreggate data yet
"""@dataclass
class travelLog:
    xLoc: int = 0
    yLoc: int = 0"""

class Walker(core.Agent): 
    TYPE = 0 
    OFFSETS = np.array([1, -1])
    
    
    #distribution = None

    # The initialization method that runs upon object initialization
    # The arguments are what I want for my custom walker.
    def __init__(self, local_object_id: int, rank: int, pt: dpt): 
        # Initialize the parent class 
        super().__init__(id=local_object_id,type=Walker.TYPE,rank=rank)
        self.pt = pt #Init the variable used for storing agent location.
        self.qualityMem =  torch.zeros(gridShape, dtype=torch.half) # initialize variable used for storing agent memory. Starts at 0
        #self.attractionMem =  torch.zeros(gridShape, dtype=torch.half) # initialize variable used for storing agent memory. Starts at 0
            # may not need to initialize this 

    def save(self) -> Tuple:
        """Saves the state of this Walker as a Tuple.
        Returns:
            The saved state of this Walker.
        """
        return (self.uid, self.pt.coordinates, self.qualityMem)

    

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
        
    def memoryWalk(self, grid, values): 
        
        # Import qualityValuesFromGrid range[0,1] or bool
        quality = values.grid
        
        # calculate distnace matrix
        perceptionMat = calcDist()
        
        #mulitply true quality with the perceptionTensor 
        quality.mul(perceptionMat)
        
        quashedExpec = (1 - np.exp(-beta*dT))*expectation #scalar 
        
        quality.add(perceptionMat.mul(-1).add(1).mul(self.qualityMem.mul(expectation).add(quashedExpec))) 
        # quality + [(-1)*perceptionMat + 1 ]*[previousqualityMem*forgetting + expectation_envelope*expectation]    

        self.qualityMem = quality

        costFunc = # is a 2D tensor
        
        quality = quality.mul(costFunc) #now is the attraction matrix. currently only supports one layer 

        probability = torch.divide(quality,torch.cumsum(quality)) 

        #choose which index to move to based on the probability kernel 

        self.pt = grid.move(self, dpt(newX,\
                                      newY,\
                                      0))

walker_cache = {} 
# this cache maintains all pointers of the created walkers in this model! (pointers don't exist in python, but I think it works like that.)
# It seems like the cache is never accessed by any other method other and restore walker. This is, in other words, more like a private variable. 
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

    if uid in walker_cache:                     # If the uid exists in the cache
        walker = walker_cache[uid]              # point to an existing walker 
    else:
        walker = Walker(uid[0], uid[2], pt)     # Else create a new
        walker_cache[uid] = walker

    walker.pt = pt
    return walker

#class World(repast4py.value_layer.ReadWriteValyeLayer):
#    def __init__(self,comm: MPI.Intracomm, params: Dict, borders: repast4py.space.BorderType, buffer_size: int, init_value: float):



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
        self.worldValuaA = repast4py.value_layer.ValueLayer(comm, box, borders=space.BorderType.Sticky, init_value = 'random')
            # init_values should be made with image 

        # Populate world with walkers
        rng = repast4py.random.default_rng
        for i in range(params['walker.count']):
            # get a random x,y location in the grid
            pt = self.grid.get_random_local_pt(rng)
            # create and add the walker to the context
            walker = Walker(i, rank, pt)
            self.context.add(walker)
            self.grid.move(walker, pt)
        
        # initialize individual logging
        self.agent_logger = logging.TabularLogger(comm, params['agent_log_file'],\
            ['tick', 'agent_id', 'agent_uid_rank', 'x', 'y'])
        
        # Count initial data at time t = 0 and log
        
    def step(self):
        for walker in self.context.agents(): 
            walker.walk(self.grid,self.worldValuaA)
        
        self.context.synchronize(restore_walker)
        #self.worldMatrix.swap_layers()
        # <WIP> Insert aggregate logging stuff. 
            #tick = self.runner.schedule.tick 
            #self.data_set.log(tick)
        # Clear temporary agregate variables for next tick

    def log_agents(self): 
        tick = self.runner.schedule.tick
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

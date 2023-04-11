##########
# Connectivity Model 
# Aaron Xu 
# v0.1.2-alpha
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

def calculateQuality(d_jt, Q, q_past, alpha, beta, expectation, dT):
        # 1st Term of eq 1. in Avgar et al 2013 
        # Mulitply true quality with the perception tensor, d_jt
        alphaDistanceDecay   = torch.exp(-alpha * d_jt / dT) #  -> tensor
        term1         = alphaDistanceDecay * Q# -> tensor
        # 2nd Term of eq 1. in Avgar et al 2013 
        # 1 - distance decay 
        oneMinusADD = (1-alphaDistanceDecay) # -> tensor 
        betaDecay      = np.exp( -beta * dT) #scalar
        oneMinusBD    = (1 - betaDecay) #scalar 
        q_new         = term1 + oneMinusADD * (betaDecay * q_past + oneMinusBD * expectation)
        return q_new

class Walker(core.Agent): 
    TYPE = 0 
    OFFSETS = np.array([1, -1])

    # The initialization method that runs upon object creation
    def __init__(self, local_object_id: int, rank: int):#, buffered_grid_shape, initMemory = None):
        """
        self:               python syntax
        local_object_id:    Unique identifier for the Walker 
        rank:               Another identifier. Tells the rank upon which the agent was created
        """ 
        # Initialize the parent class 
        super().__init__(id=local_object_id,type=Walker.TYPE,rank=rank)

        # Initialize internal variables 
        self.qualityMem = None          # Serves as a place holder right after init.
        self.cumProb = 0
        # Initialize Cognition Parameters. Implemented this way to allow these parameters to be changed, perhaps to model changes in cognition at different life stages
        self.alpha = ALPHA
        self.beta  = BETA
        self.gama = GAMA
        self.dT = DT
        self.expectation = EXPECTATION
        self.omega = OMEGA
        self.global_loc = None
    def save(self) -> Tuple:
        """Saves the state of this Walker as a Tuple.
        Returns:
            The saved state of this Walker.
        """
        return ([self.uid])
    

    def rndwalk(self, grid):
        # sample from a distribution for x and y
        # move in those directions 

        np.random.seed()
        x_dirs = round(np.random.normal(0.0, DMAX))
        y_dirs = round(np.random.normal(0.0, DMAX))

        self.pt = grid.move(self, dpt(grid.get_location(self).x + x_dirs,\
                                      grid.get_location(self).y + y_dirs,\
                                      0))
    
   
    def _update_Global_Location(self,grid,gridValue):
            ####################################################
            # Updates the global location of the agent, 
            #    which the agent carries as an internal variable. 
            #  - Does not work as inteded
            #  - May not be the best solution
            ##############################
        x_offset = gridValue.buffered_bounds.xmin 
        y_offset = gridValue.buffered_bounds.ymin
        global_location_X = grid.get_location(self).x + x_offset
        global_location_Y = grid.get_location(self).y + y_offset

        self.global_loc = dpt(global_location_X,global_location_Y,0)

    def memoryWalk(self, worldgrid: space.SharedGrid, value_layer : repast4py.value_layer.SharedValueLayer): 
        '''
        self:           python syntax
        worldgrid:      grid object that handels agents' locations 
        qualitvalues:   envrionmental values
        '''
        
        # Consume Nutrients 
        selfCoord = worldgrid.get_location(self)
        setgetCoord = dpt(selfCoord.y,selfCoord.x) # Coordinates have to be inverted 
        currentValue = value_layer[0].get(setgetCoord)

       
        if currentValue - 0.222 < 0:
            newValue = currentValue * 0.5
        else:
             newValue = currentValue-0.222
        value_layer[0].set(setgetCoord,newValue)

        
        gridShape = value_layer[0].grid.shape

        #Populate qualityMem upon use. Done this way to allow the size of qualityMem to 
        #   depend on local gridshapes which may vary from rank to rank. Since 
        #   0.1.2-alpha does not support parallel processing, this doesn't matter. 
        if self.qualityMem == None:
            self.qualityMem = torch.zeros(gridShape, dtype=torch.float)
        
        # Copy qualityvalues
        quality = torch.cat((value_layer[0].grid.expand(1,-1,-1),value_layer[1].grid.expand(1,-1,-1)),0)
        
        # get agent location
        local_X =  worldgrid.get_location(self).x % gridShape[0]
        local_Y = worldgrid.get_location(self).y % gridShape[1]

        #~~~~ Begin Cognition Calculations ~~~~# 
        # See Thesis or Avgar et al. 
        # Calculate preception Matrix: Each element of the matrix is the euclidean distance to the element (local_Y, Local_X)
        perceptionMat = mk_insert_distance.pasteKernel([local_X, local_Y], 
                                                        len(KERNEL),\
                                                        gridShape, \
                                                        KERNEL)
        costFuncGamma  = torch.exp(-self.gama * perceptionMat / self.dT)# -> 2D tensor
        costFuncGamma[costFuncGamma < 0.01] = 0.0

        self.qualityMem = calculateQuality(perceptionMat,quality,self.qualityMem,self.alpha,self.beta,self.expectation,self.dT).clone()
        
        qualityToOmega = torch.pow(self.qualityMem,self.omega) #omega must be in shape (len(qualityMem),1,1)

        for idx, layer in enumerate(qualityToOmega):
            if idx == 0:
                attraction = layer
            else:
                attraction = attraction * layer

        attraction = costFuncGamma * attraction #now is the attraction matrix. currently only supports one layer 
        attraction[attraction.isinf()] = 0
        attraction[attraction.isneginf()] = 0
        attraction[attraction.isnan()] = 0
        #attraction      = torch.clip(attraction, min = 0, max = 1) # make sure no numbers go out of range
        #   attraction[attraction < ] = 0 #ignore attractions that are below a certain magnitude 
        
        # In an edge case where the total attraction is exactly zero, perform a random walk. Ideally this never runs since
        #   the default expectation is non-zero
        if torch.sum(attraction) == 0.: 
            print("Random Walk")
            self.rndwalk(worldgrid)
        else:
            probability = attraction / torch.sum(attraction)
            #probability[probability.isinf()] = 0. # ensure no infinities appear
            #print(probability[probability.isinf()])
            # In order to chose a coordinate based on redistribution probability using numpy,
            #   the probability tensor must be flattened.
            probFlat =  probability.flatten().numpy()
            try:
                idx1D = np.random.choice(a = len(probFlat), size = 1, replace = True, p = probFlat)  #<- choose which index to move to based on the probability kernel 
            except:
                plt.figure(1)
                plt.imshow(self.qualityMem[0])
                plt.figure(2)
                plt.imshow(self.qualityMem[1])
                plt.figure(3)
                plt.imshow(qualityToOmega[0])
                
                plt.figure(3)
                plt.imshow(qualityToOmega[1])
                
                plt.figure(4)
                plt.imshow(attraction)
                plt.show()
                print(attraction[local_X,local_Y])
            dim1Size = quality.size(dim=1)
            
            # If the 1D index starts with zero, the algorithm to recover its unflattened index is 
            local_newX = int(idx1D[0]/ dim1Size)
            local_newY = idx1D[0] % dim1Size

            """ # Section Meant to handle parallel processes and rank-local vs global locations 
            delta_X = local_newX-local_X
            delta_Y = local_newY-local_Y
            
            world_newX = worldgrid.get_location(self).x+delta_X
            world_newY = worldgrid.get_location(self).y+delta_Y"""
            
            self.pt = worldgrid.move(self, dpt(local_newX,
                                               local_newY,0))
            #print(float(probFlat[idx1D]))
            self.cumProb += np.log(float(probFlat[idx1D]))
            # Was supposed to update the agent's global position when using parallel processing. 
            self._update_Global_Location(worldgrid,value_layer[0])

    

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
    if uid in walker_cache:                     # If the uid exists in the cache
       #print("@restore_walker - Restoring to Hold Skin/husk")
        walker = walker_cache[uid]              # point to an existing walker 
    else:
        #print("@restore_walker - Create a new husk")
        walker = Walker(uid[0], uid[2])  
        walker_cache[uid] = walker
    
    return walker

def initialize_grid_values(width,height,seed,density = 0.0): 
    
    baseFoodVals = map_generation.makePerlinRaster(width,height,seed)
    print("DONE : baseFoodVals")
    baseRiskVals = map_generation.makePerlinRaster(width,height,seed*4)
    print("DONE : baseRiskVals")
    regionBoundaries = np.array([[0, height-1, round(width*(1/3)), round(width*(2/3))]])
    regionMasks = map_generation.make_region_map(regionBoundaries,width,height)
    print("DONE : region masks")
    regionWithDiracDeltas = map_generation.select_random_pts_region(density,regionMasks)
    maskedFoodVals = baseFoodVals*(1-regionWithDiracDeltas[0])
    print("DONE : maskedFoodVals")
    maskedRiskVals = torch.clip(baseRiskVals+(1.0 * regionWithDiracDeltas[0]),min=0,max=1)
    print("DONE : maskedRiskVals")
    return torch.cat((maskedFoodVals.expand(1,-1,-1), maskedRiskVals.expand(1,-1,-1)), 0)

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
    def __init__(self, comm: MPI.Intracomm, params: Dict,layers): 
        
        
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n Initializing Model ... \n")
        
        
        self.onRankFood = layers[0]#[localBounds.ymin:(localBounds.ymin+localBounds.yextent),
                                #localBounds.xmin:(localBounds.xmin+localBounds.xextent)]
        np.savetxt(params['enviro_file']+"_food.csv", self.onRankFood.numpy(),delimiter=',')
        self.onRankRisk = layers[1]#[localBounds.ymin:(localBounds.ymin+localBounds.yextent),
                               # localBounds.xmin:(localBounds.xmin+localBounds.xextent)]
        np.savetxt(params['enviro_file']+"_risk_.csv", self.onRankRisk.numpy(),delimiter=',')
        # Create the schedule 
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1,1,self.step)
        self.runner.schedule_repeating_event(730,730,self.envrioStep) # number of ticks for flower recharge 
        self.runner.schedule_repeating_event(1.1,1,self.log_agents) 
        
        # TEMPLATE: self.runner.schedule_repeating_event(<when to start>,<how often>,self.<method>)
       
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
                                     occupancy=space.OccupancyType.Multiple, buffer_size=DMAX, comm=comm) # buffersize should be DMAX
        self.context.add_projection(self.grid)      
        self.rank = comm.Get_rank()
        self.nproc = comm.Get_size()      
        print(["@Init - Rank", self.rank,", Num Ranks: ", self.nproc])
    
        # Get Local Bounds to divie up the raster values into chunks for parallel processing. 
        # NOTE: Paralel processing note supported in v0.1.2-apha
        localBounds = self.grid.get_local_bounds()
        print(["@Init - Local Bounds:", localBounds])
        
        # initializing value_layer objects 
        # For Future: init_values could be made with image 
        self.foodValue = repast4py.value_layer.SharedValueLayer(comm, box, borders=space.BorderType.Sticky, buffer_size = DMAX
         ,init_value = self.onRankFood) # Buffersize here should depend on alpha, not DMAX
        
        plt.imshow(self.foodValue.grid)
        plt.title(self.rank)
        plt.show()

        print(self.foodValue.buffered_bounds)
        
        self.riskValue = repast4py.value_layer.SharedValueLayer(comm, box, borders=space.BorderType.Sticky, buffer_size = DMAX
         ,init_value = self.onRankRisk)
        
        # Populate world with walkers
        rng = np.random.default_rng(seed = params['random.seed'])
        for i in range(params['walker.count']):
            # get a random x,y location in the grid
            #pt = self.grid.get_random_local_pt(rng)
            pt = dpt(rng.choice(range(0,400)),rng.choice(range(0,133)))
            # create and add the walker to the context
            #walker = Walker(np.random.choice([0,1,2,3,4,5,6,7]), self.rank, pt, buffered_grid_shape=self.foodValue.grid.shape)
            walker = Walker(i, self.rank) #, buffered_grid_shape=self.foodValue.grid.shape)
            self.context.add(walker)
            self.grid.move(walker, pt)
            walker._update_Global_Location(self.grid,self.foodValue)
        
        # initialize individual logging
        self.agent_logger = logging.TabularLogger(comm, params['agent_log_file'],\
                                                        ['tick', 'agent_id', 'agent_uid_rank', 'x', 'y','log_cum_prob'])
        
        
    def step(self):
        #print(["@step - ", "Rank", self.rank, "Num Agents",self.context.size()])
        #print("@step - ", self.t)
        for walker in self.context.agents(): 
            walker.memoryWalk(self.grid,[self.foodValue,self.riskValue])
        
        self.context.synchronize(restore_walker)    
        

        # Highly inneficcient way to update every foodValue Cell
        """for i in range(self.foodValue.grid.shape[0]):
            for j in range(self.foodValue.grid.shape[1]):
                growthPoint = dpt(i,j)
                self.foodValue.set(growthPoint,self.foodValue.get(growthPoint)*1.0001)"""
        #self.worldMatrix.swap_layers()
        # <WIP> Insert aggregate logging stuff. 
            #tick = self.runner.schedule.tick 
            #self.data_set.log(tick)
        # Clear temporary agregate variables for next tick
    def envrioStep(self):
        # Highly inneficcient way to update every foodValue Cell
        for i in range(self.foodValue.grid.shape[0]):
            for j in range(self.foodValue.grid.shape[1]):
                growthPoint = dpt(i,j)
                self.foodValue.set(growthPoint, 0.1*self.foodValue.get(growthPoint)**(1/2)+\
                                                0.9*self.foodValue.get(growthPoint)) #the operetation was an august idea
                self.foodValue.set(growthPoint, self.foodValue.get(growthPoint)**(1/2)*\
                                                self.onRankFood[j,i])
    def log_agents(self): 
        tick = self.runner.schedule.tick
        if round(tick)%100 == 0:
            print("@Log - tick: ", tick)
        for walker in self.context.agents():
            self.agent_logger.log_row(tick, walker.id,walker.uid_rank,walker.global_loc.x ,walker.global_loc.y, walker.cumProb)
        self.agent_logger.write() # not necessary to call every time. Potential for optimization by deciding how often to write
    
    def at_end(self):
        # self.data_set.close() # commented out because no aggregate data collection yet
        """for walker in self.context.agents():
            plt.figure(1)
            plt.imshow(walker.qualityMem[0])
            plt.figure(2)
            plt.imshow(walker.qualityMem[1])
            plt.show()"""
        plt.imshow(self.foodValue.grid)
        plt.show()
        self.agent_logger.close()
    
    def start(self):
        self.runner.execute() 

def run(params: Dict):
    # Initialize globally accessible cognition parameters
    global DT 
    global GAMA
    global ALPHA
    global BETA 
    global DMAX
    global EXPECTATION 
    global KERNEL
    global OMEGA

    DT = params['time.step']
    GAMA = params['gamma.type0']
    ALPHA = params['alpha.type0']
    BETA = params['beta.type0']
    DMAX = -round(DT * np.log(0.01) / GAMA) 
    EXPECTATION = params['expectation.type0']
    omega0 = params['omega0']
    omega1 = params['omega1']
    # omegax = params['omegax']
    OMEGA = torch.tensor([[[omega0]],[[omega1]]])

    print("@ run: - DMAX", DMAX)

    KERNEL = mk_insert_distance.makeDistanceKernel(int(DMAX),DMAX)
    print("Initializing Rasters ...")
    rasters = initialize_grid_values(params['world.width'], params['world.height'],params['random.seed'],density=params['matrix.density']) # can source the raster from other places     
    model = Model(MPI.COMM_WORLD, params,rasters)
    model.start()

if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    params = parameters.init_params(args.parameters_file, args.parameters)
    run(params)

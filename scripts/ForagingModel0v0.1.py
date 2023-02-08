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

class Walker(core.Agent): 
    Type = 0 
    distribution = None

    # The initialization method that runs upon object initialization
    # The arguments are what I want for my custom walker.
    def __init__(self, local_object_id: int, rank: int, pt: dpt): 
        # Initialize the parent class 
        super().__init__(id=local_object_id,type=Walker.Type,rank=rank)
        self.pt = pt #Init the variable used for storing agent location.
    
    def save(self) -> Tuple:
        """Saves the state of this Walker as a Tuple.
        Returns:
            The saved state of this Walker.
        """
        return (self.uid, self.pt.coordinates)

    def walk(self, grid): 
        # sample from a distribution for theta and r 
        # convert to x and y coordinates 
        # move in those directions 

        # <WIP Code> 

        self.pt = grid.move(self, dpt(self.pt.x + xy_dirs[0],\
                                      self.pt.y + xy_dirs[1],\
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
        # self.runner.schedule_repeating_event(<when to start>,<how often>,self.<method>)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end) 

        # create the context to hold the agents and manage cross process synchronization
        self.context = ctx.SharedContext(comm) 

        # create a bounding box equal to the size of the entire global world grid
        box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
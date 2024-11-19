import torch
from torch.distributions import Categorical
import random
import os
import itertools
import numpy as np
from functools import cached_property
import concurrent.futures
from itertools import combinations
import time 
from concurrent.futures import ProcessPoolExecutor

CAPACITY = 1.0 # The input demands shall be normalized

class search_route():
    def __init__(
        self,  # 0: depot
        distances, # (n, n)
        demand,   # (n, )
        genetate=20, 
        heuristic=None,
        heuristic_target=None,
        device='cpu',
        capacity=CAPACITY,
        positions = None,
    ):
        
        self.problem_size = len(distances)
        self.distances = distances
        self.capacity = capacity
        self.demand = demand
        self.n_genetate = genetate
        self.positions = positions
        self.heuristic = heuristic
        self.heuristic_target = heuristic_target
        self.shortest_path = None
        self.lowest_cost = float('inf')
        self.device = device

    @cached_property
    @torch.no_grad()   
    def selection(self, population, fitnesses):
        fitnesses1 = 1 / (np.array(fitnesses) + 1e-5)
        total_fitness = np.sum(fitnesses1)
        probabilities = fitnesses1 / total_fitness
        cumulative_probabilities = np.cumsum(probabilities)
        selected_indices = []
        for _ in range(2):
            r = random.random()
            for i, cum_prob in enumerate(cumulative_probabilities):
                if r < cum_prob:
                    selected_indices.append(i)
                    break
        return population[selected_indices[0]], population[selected_indices[1]]

    @torch.no_grad()
    def test(self, n_iterations, time_limit):
        paths = self.generate_r(require_prob=False)
        rows, cols = paths.shape
        if rows < int(self.problem_size * 1.5):
            num_zeros_to_add = int(self.problem_size * 1.5) - rows
            zeros_to_add = torch.zeros((num_zeros_to_add, cols), dtype=paths.dtype).to(self.device)
            paths = torch.cat((paths, zeros_to_add), dim=0)
        #print(paths.size(0),paths.size(1))
        costs = self.get_cost(paths)
        best_cost, best_idx = costs.min(dim=0)
        self.shortest_path = paths[:, best_idx].clone()
        self.lowest_cost = best_cost.item()
        return self.lowest_cost, 0
   
    def done_new(self, visit_mask, actions,desi=0):
      if desi==0:
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()
      else:
        return (visit_mask[:, 1:] == 0).all()
      
    @torch.no_grad()
    
    def next_action(self, dist, visit_mask,  require_prob, invtemp=1.0, selected=None,capacity_mask=None,desi=0):
        if desi==0:
          dist = (dist ** invtemp) * visit_mask * capacity_mask  # shape: (n_genetate, p_size)
          #if random.uniform(0,1)<0.01:
              #dist = (torch.ones_like(dist)**invtemp) * visit_mask * capacity_mask
        if desi==1:
          dist = (dist ** invtemp) #* visit_mask * capacity_mask  # shape: (n_genetate, p_size)
        dist = dist / dist.sum(dim=1, keepdim=True)  # This should be done for numerical stability
        #print(dist,selected)
        flag=0
        if (dist.max(1)[0].mean()-dist.mean(1).mean())/dist.mean(1).mean()<130:#380#240#150#140
            flag=1
        #print( (dist.max(1)[0].mean()-dist.mean(1).mean())/dist.mean(1).mean())
        dist = Categorical(probs=dist,validate_args=False)
        #if random.uniform(0,1)<0.7:
        if selected is not None:
            actions = selected
        else:
            actions = dist.probs.argmax(dim=-1) 
        
        if random.uniform(0,1)<0.05:#flag==1:#random.uniform(0,1)<0.2:#0.05:#0.2
         actions = selected if selected is not None else dist.sample()  # shape: (n_genetate,)
        log_probs = dist.log_prob(actions) if require_prob else None  # shape: (n_genetate,)
        return actions, log_probs
    
    def get_cost(self, paths):
        u = paths.permute(1, 0) # shape: (n_genetate, max_seq_len)
        v = torch.roll(u, shifts=-1, dims=1)  

        return torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)  
    def visit_mask_new(self, visit_mask, actions,_genetate):
        visit_mask[torch.arange(_genetate, device=self.device), actions] = 0
        visit_mask[:, 0] = 1 # depot can be revisited with one exception
        visit_mask[(actions==0) * (visit_mask[:, 1:]!=0).any(dim=1), 0] = 0 # one exception is here
        return visit_mask
    
    def capacity_mask_new(self, cur_nodes, used_capacity):

        capacity_mask = torch.ones(size=(self.n_genetate, self.problem_size), device=self.device)
        # update capacity
        used_capacity[cur_nodes==0] = 0
        used_capacity = used_capacity + self.demand[cur_nodes]
        # update capacity_mask
        remaining_capacity = self.capacity - used_capacity # (n_genetate,)
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, self.problem_size) # (n_genetate, p_size)
        demand_repeat = self.demand.unsqueeze(0).repeat(self.n_genetate, 1) # (n_genetate, p_size)
        capacity_mask[demand_repeat > remaining_capacity_repeat + 1e-10] = 0
        
        return used_capacity, capacity_mask
    

    def generate_r(self, require_prob=False, invtemp=1.0, paths=None, desi=0):
        if desi == 0:
            num_generate = self.n_genetate
        else:
            num_generate = 1

        actions = torch.zeros((num_generate,), dtype=torch.long, device=self.device)
        visit_mask = torch.ones(size=(num_generate, self.problem_size), device=self.device)
        visit_mask = self.visit_mask_new(visit_mask, actions, num_generate)
        capacity_mask = None

        if desi == 0:
            used_capacity = torch.zeros(size=(num_generate,), device=self.device)
            used_capacity, capacity_mask = self.capacity_mask_new(actions, used_capacity)

        prob_mat = (torch.ones_like(self.distances)) * (self.heuristic)
        prev_actions = actions

        paths_list = [actions]
        log_probs_list = []
        done = self.done_new(visit_mask, actions)

        # Given paths
        step_index = 0
        if desi == 1:
            step_index = -1
        feasible_idx = torch.arange(num_generate, device=self.device) if paths is not None else None

        while not done:
            selected_action = paths[step_index + 1] if paths is not None else None
            actions, log_probs = self.next_action(
                dist=prob_mat[prev_actions],
                visit_mask=visit_mask,
                capacity_mask=capacity_mask,
                require_prob=require_prob,
                invtemp=invtemp,
                selected=selected_action,
                desi=desi
            )
            paths_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                visit_mask = visit_mask.clone()
            visit_mask = self.visit_mask_new(visit_mask, actions, num_generate)
            if desi == 0:
                used_capacity, capacity_mask = self.capacity_mask_new(actions, used_capacity)

            done = self.done_new(visit_mask, actions, desi)
            prev_actions = actions
            step_index += 1

        if require_prob:
            if paths is not None:
                return torch.stack(log_probs_list)
            return torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            return torch.stack(paths_list)

    
    








import torch
from torch.distributions import Categorical
import random
import itertools
import numpy as np
from swapstar import get_better
from functools import cached_property
import concurrent.futures
from itertools import combinations


CAPACITY = 1.0 # The input demands shall be normalized



class search():
    def __init__(
        self,  # 0: depot
        distances, # (n, n)
        demand,   # (n, )
        generate=20, 
        alpha=1,
        beta=1,
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
        
        self.generate = generate
        self.alpha = alpha
        self.beta = beta
        self.positions = positions

        self.heuristic = 1 / (distances + 1e-10) if heuristic is None else heuristic
        self.heuristic_target = 1 / (distances + 1e-10) if heuristic_target is None else heuristic_target
        self.shortest_path = None
        self.lowest_cost = float('inf')

        self.device = device

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
        
        if random.uniform(0,1)<1.1:#flag==1:#random.uniform(0,1)<0.2:#0.05:#0.2
         actions = selected if selected is not None else dist.sample()  # shape: (n_genetate,)
        log_probs = dist.log_prob(actions) if require_prob else None  # shape: (n_genetate,)
        return actions, log_probs


    @torch.no_grad()
    def vali_(self, n_iterations):
        for _ in range(n_iterations):
            paths = self.generate_r(require_prob=False)
            costs = self.get_costs(paths)
            #print(costs)
            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx].clone()  # type: ignore
                self.lowest_cost = best_cost.item()

        return self.lowest_cost, 0

    def visit_all(self, visit_mask, actions):
        visit_mask[torch.arange(self.generate, device=self.device), actions] = 0
        visit_mask[:, 0] = 1 # depot can be revisited with one exception
        visit_mask[(actions==0) * (visit_mask[:, 1:]!=0).any(dim=1), 0] = 0 # one exception is here
        return visit_mask
    
    def capacity_all(self, cur_nodes, used_capacity):

        capacity_mask = torch.ones(size=(self.generate, self.problem_size), device=self.device)
        # update capacity
        used_capacity[cur_nodes==0] = 0
        used_capacity = used_capacity + self.demand[cur_nodes]
        # update capacity_mask
        remaining_capacity = self.capacity - used_capacity # (generate,)
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, self.problem_size) # (generate, p_size)
        demand_repeat = self.demand.unsqueeze(0).repeat(self.generate, 1) # (generate, p_size)
        capacity_mask[demand_repeat > remaining_capacity_repeat + 1e-10] = 0
        
        return used_capacity, capacity_mask
    
    def done_all(self, visit_mask, actions):
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()
    
    def generate_route(self, invtemp=1.0):
        paths, log_probs = self.generate_r(require_prob=True, invtemp=invtemp)  # type: ignore
        paths, log_probs_D = self.generate_r(require_prob=True, invtemp=invtemp,paths=paths) 
        costs = self.get_costs(paths)
        return costs, log_probs, paths,log_probs_D
    
    @torch.no_grad()
    def get_costs(self, paths):
        u = paths.permute(1, 0) # shape: (generate, max_seq_len)
        v = torch.roll(u, shifts=-1, dims=1)  
        return torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)

    def generate_r(self, require_prob=False, invtemp=1.0, paths=None):
        actions = torch.zeros((self.generate,), dtype=torch.long, device=self.device)
        used_capacity = torch.zeros(size=(self.generate,), device=self.device)
        used_capacity, capacity_mask = self.capacity_all(actions, used_capacity)
        visit_mask = torch.ones(size=(self.generate, self.problem_size), device=self.device)
        visit_mask = self.visit_all(visit_mask, actions)
        if paths==None:
            prob_mat = (torch.ones_like(self.distances) ** self.alpha) * (self.heuristic ** self.beta)
        else:
            prob_mat = (torch.ones_like(self.distances) ** self.alpha) * (self.heuristic_target ** self.beta)
        prev = actions

        paths_list = [actions]  # paths_list[i] is the ith move (tensor) 
        log_probs_list = []  # log_probs_list[i] is the ith log_prob (tensor) 
        done = self.done_all(visit_mask, actions)

        ##################################################
        # given paths
        i = 0
        feasible_idx = torch.arange(self.generate, device=self.device) if paths is not None else None
        ##################################################
        while not done:
            selected = paths[i + 1] if paths is not None else None
            actions, log_probs = self.next_action(prob_mat[prev], visit_mask, require_prob,  invtemp, selected,capacity_mask)
            paths_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                visit_mask = visit_mask.clone()
            visit_mask = self.visit_all(visit_mask, actions)
            used_capacity, capacity_mask = self.capacity_all(actions, used_capacity)

            done = self.done_all(visit_mask, actions)
            prev = actions
            i += 1

        if require_prob:
            if paths is not None:
                return torch.stack(paths_list), torch.stack(log_probs_list)#, feasible_idx  # type: ignore
            return torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            return torch.stack(paths_list)


    


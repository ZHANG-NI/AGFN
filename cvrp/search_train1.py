import torch
from torch.distributions import Categorical
import random
import itertools
import numpy as np
from swapstar import get_better
from functools import cached_property
import concurrent.futures
from itertools import combinations
#from alns1 import alns_use
from LKH_path import generate_lkh
CAPACITY = 1.0 # The input demands shall be normalized


class search1():
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
               
        self.heuristic =  heuristic
        self.heuristic_target = heuristic_target
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

    def generate_route(self, invtemp=1.0):
        paths, log_probs = self.generate_r(require_prob=True, invtemp=invtemp)  # type: ignore
        self.get_better_result(paths, inference=True)
        paths, log_probs = self.generate_r(require_prob=True, invtemp=invtemp, paths=paths, desi=1)
        costs = self.get_costs(paths)
        return costs, log_probs, paths
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
    
    def done_new(self, visit_mask, actions):
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()
    @torch.no_grad()
    def get_better_result(self, paths, indexes=None, inference=False):
        subroutes_all = []
        for i in range(paths.size(1)) if indexes is None else indexes:
            subroutes = div_routes(paths[:, i])
            subroutes_all.append((i, subroutes))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i, p in subroutes_all:
                future = executor.submit(
                    better,
                    self.demand_cpu,
                    self.distances_cpu,
                    self.heuristic_dist,
                    self.positions_cpu,
                    p,
                    limit=10000 if inference else self.problem_size // 10
                )
                futures.append((i, future))
            for i, future in futures:
                paths[:, i] = comb_routes(future.result(), paths.size(0), self.device)

    @cached_property
    @torch.no_grad()
    def heuristic_dist(self):
        heu = self.heuristic.detach().cpu().numpy()  # type: ignore
        return heu
        #return (1 / (heu/heu.max(-1, keepdims=True) + 1e-5))
    
    @torch.no_grad()
    def get_costs(self, paths):
        u = paths.permute(1, 0) # shape: (generate, max_seq_len)
        v = torch.roll(u, shifts=-1, dims=1)  
        return torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)

    def generate_r(self, require_prob=False, invtemp=1.0, paths=None,desi=0):
        #paths=generate_lkh(self.heuristic,self.demand,self.distances)
        used_capacity = torch.zeros(size=(self.generate,), device=self.device)
        actions = torch.zeros((self.generate,), dtype=torch.long, device=self.device)
        used_capacity, capacity_mask = self.capacity_all(actions, used_capacity)
        visit_mask = torch.ones(size=(self.generate, self.problem_size), device=self.device)
        visit_mask = self.visit_all(visit_mask, actions)

        prob_mat = (torch.ones_like(self.distances)** self.alpha) * (self.heuristic** self.beta)
        if desi==1:
            prob_mat = (torch.ones_like(self.distances)** self.alpha) * (self.heuristic_target** self.alpha )
        prev = actions

        paths_list = [actions]  # paths_list[i] is the ith move (tensor) for all generate
        log_probs_list = []  # log_probs_list[i] is the ith log_prob (tensor) for all generate' actions
        done = self.done_new(visit_mask, actions)

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

            done = self.done_new(visit_mask, actions)
            prev = actions
            i += 1
        while len(paths_list)<300:
            #pdb.set_trace()
            actions = torch.zeros(len(paths_list[0]), dtype=torch.int).to(self.device)
            paths_list.append(actions)
        if require_prob:
            if paths is not None:
                return torch.stack(paths_list), torch.stack(log_probs_list)  # type: ignore
            return torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            return torch.stack(paths_list)

    
    @cached_property
    @torch.no_grad()
    def distances_cpu(self):
        return self.distances.cpu().numpy()
    
    @cached_property
    @torch.no_grad()
    def demand_cpu(self):
        return self.demand.cpu().numpy()
    
    @cached_property
    @torch.no_grad()
    def positions_cpu(self):
        return self.positions.cpu().numpy() if self.positions is not None else None
    


def better(demand, distances, heu_dist, positions, p, disturb=5, limit=10000):
    p0 = p
    p1 = get_better(demand, distances, positions, p0, count = limit)
    p2 = get_better(demand, heu_dist, positions, p1, count = disturb)
    p3 = get_better(demand, distances, positions, p2, count = limit)
    return p3

def div_routes(route, end_with_zero=True):
    indices = torch.nonzero(route == 0).flatten()
    subroutes = [route[start:end + (1 if end_with_zero else 0)] for start, end in zip(indices, indices[1:]) if end - start > 1]
    return subroutes


def comb_routes(subroutes, length, device):
    route = torch.zeros(length, dtype=torch.long, device=device)
    i = 0
    for r in subroutes:
        if len(r) > 2:
            r = torch.tensor(r[:-1], dtype=torch.long, device=device) if isinstance(r, list) else r[:-1]
            route[i:i + len(r)] = r
            i += len(r)
    return route
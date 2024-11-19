import torch
import numpy as np
import numba as nb
from torch.distributions import Categorical
from two_opt import batched_two_opt_python
import random
import concurrent.futures
from functools import cached_property
from itertools import combinations

#python test.py 200 -p 
class search_():
    def __init__(
        self, 
        distances,
        generate=20, 
        alpha=1,
        beta=1,
        heuristic=None,
        heuristic_target=None,
        two_opt=False,
        device='cuda:0',
        local_search: str | None = 'nls',
    ):

        self.scale_ = len(distances)
        self.distances = distances.to(device)
        self.generate = generate
        self.alpha = alpha
        self.beta = beta
        assert local_search in [None, "2opt", "nls"]
        self.local_search_type = '2opt' if two_opt else local_search

        self.heuristic = 1 / (distances + 1e-10) if heuristic is None else heuristic
        self.heuristic_target = 1 / (distances + 1e-10) if heuristic_target is None else heuristic_target
        self.shortest_path = None
        self.lowest_cost = float('inf')

        self.device = device

    def get_route(self, alpha=1.0, require_prob=False, paths=None, start_node=None,desi=0):
        if paths is None:
            if start_node is None:
                start = torch.randint(low=0, high=self.scale_, size=(self.generate,), device=self.device)
            else:
                start = torch.ones((self.generate,), dtype = torch.long, device=self.device) * start_node
        else:
            start = paths[0]

        mask = torch.ones(size=(self.generate, self.scale_), device=self.device)
        index = torch.arange(self.generate, device=self.device)
        if paths==None:
            prob_mat = (torch.ones_like(self.distances).to(self.device) ** self.alpha) * (self.heuristic ** self.beta)
        else:
            prob_mat = (torch.ones_like(self.distances).to(self.device) ** self.alpha) * (self.heuristic_target ** self.beta)
        mask[index, start] = 0

        paths_list = [] 
        paths_list.append(start)

        log_probs_list = []  
        prev = start
        for i in range(self.scale_ - 1):
            dist = (prob_mat[prev] ** alpha) * mask
            dist = dist / dist.sum(dim=1, keepdim=True)  
            dist = Categorical(probs=dist)
            epsilon =0.05
            if desi==1:
                actions = torch.where(torch.rand_like(dist.probs[:,0]) < epsilon, dist.sample(), dist.probs.argmax(dim=-1))
            else:
                actions = paths[i + 1] if paths is not None else dist.sample()
            paths_list.append(actions)
            if require_prob:
                log_probs = dist.log_prob(actions) 
                log_probs_list.append(log_probs)
                mask = mask.clone()
            prev = actions
            mask[index, actions] = 0

        if require_prob:
            return torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            return torch.stack(paths_list)
    def generate_route(self, alpha=1.0, inference=False, start_node=None):
        paths, log_probs = self.get_route(alpha=alpha, require_prob=True, start_node=start_node)
        paths, log_probs_D = self.get_route(alpha=alpha, require_prob=True, start_node=start_node,paths=paths)
        costs = self.get_costs(paths)
        return costs, log_probs, paths,log_probs_D

    def improve_route(self, paths,start_node=None,alpha=1.0):
        paths = self.improve(paths)
        costs = self.get_costs(paths)
        paths, log_probs_D = self.get_route(alpha=alpha, require_prob=True, start_node=start_node,paths=paths)
        return costs, paths, log_probs_D

    def improve(self, paths, inference=False):
        if self.local_search_type == "2opt":
            paths = self.improve_t(paths, inference)
        elif self.local_search_type == "nls":
            paths = self.improve_n(paths, inference)
        return paths

    @torch.no_grad()
    def val(self, n_iterations, inference=True, start_node=None):
        assert n_iterations > 0

        for _ in range(n_iterations):
            paths = self.get_route(alpha=1.0, require_prob=False, start_node=start_node,desi=1)
            _paths = paths.clone()  
            costs = self.get_costs(paths)

            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost.item()

        return self.lowest_cost, 0


    @torch.no_grad()
    def get_costs(self, paths):
        assert paths.shape == (self.scale_, self.generate)
        u = paths.T 
        v = torch.roll(u, shifts=1, dims=1)  
        return torch.sum(self.distances[u, v], dim=1)

    def gen_numpy_path_costs(self, paths):
        assert paths.shape == (self.generate, self.scale_)
        u = paths
        v = np.roll(u, shift=1, axis=1)  
        return np.sum(self.distances_numpy[u, v], axis=1)

    

    @cached_property
    def distances_numpy(self):
        return self.distances.detach().cpu().numpy().astype(np.float32)

    @cached_property
    def heuristic_numpy(self):
        return self.heuristic.detach().cpu().numpy().astype(np.float32) 

    @cached_property
    def heuristic_dist(self):
        return 1 / (self.heuristic_numpy / self.heuristic_numpy.max(-1, keepdims=True) + 1e-5)

    def improve_t(self, paths, inference = False):
        maxt = 10000 if inference else self.scale_ // 4
        best_paths = batched_two_opt_python(self.distances_numpy, paths.T.cpu().numpy(), max_iterations=maxt)
        best_paths = torch.from_numpy(best_paths.T.astype(np.int64)).to(self.device)

        return best_paths

    def improve_n(self, paths, inference=False, T_nls=5, T_p=20):
        maxt = 10000 if inference else self.scale_ // 4
        best_paths = batched_two_opt_python(self.distances_numpy, paths.T.cpu().numpy(), max_iterations=maxt)
        best_costs = self.gen_numpy_path_costs(best_paths)
        new_paths = best_paths

        for _ in range(T_nls):
            perturbed_paths = batched_two_opt_python(self.heuristic_dist, new_paths, max_iterations=T_p)
            new_paths = batched_two_opt_python(self.distances_numpy, perturbed_paths, max_iterations=maxt)
            new_costs = self.gen_numpy_path_costs(new_paths)

            improved_indices = new_costs < best_costs
            best_paths[improved_indices] = new_paths[improved_indices]
            best_costs[improved_indices] = new_costs[improved_indices]

        best_paths = torch.from_numpy(best_paths.T.astype(np.int64)).to(self.device)
        return best_paths


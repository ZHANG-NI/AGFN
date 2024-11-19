import os
import random
import time

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from net import Net
from search_test import search_route
from utils import load_test_dataset

from typing import Tuple, List

##python test_cvrp.py 200 -p 
EPS = 1e-10

def divide_routes(generate_r):
    r_process = torch.nonzero(generate_r == 0).flatten()
    divroutes = [generate_r[m:n+1] for m, n in zip(r_process, r_process[1:]) if n - m > 1]
    return divroutes

@torch.no_grad()
def main(ckpt, nodes, k_sparse, generate=100, iter=10000, time=1.0):
    test_list = load_test_dataset(nodes, k_sparse, DEVICE, False)
    test_list = test_list#[:(size or len(test_list))]
    t = list(range(1, 2))
    net = Net(gfn=True, Z_out_dim=1).to(DEVICE)
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    avg_cost, avg_diversity, duration = result(test_list, net, generate, t,iter,time)
    print('time_total: ', duration)
    print(f"avg. cost {avg_cost[0]}, avg. diversity {avg_diversity[0]}")


def result(all_data, model, generate, t, iter, time_limit):
    begin = time.time()
    _t = [0] + t
    t_diff = [_t[i + 1] - _t[i] for i in range(len(_t) - 1)]

    results_total = torch.zeros(size=(len(t_diff),))
    diversities_total = torch.zeros(size=(len(t_diff),))
    t=0
    for pyg_data, demands, distances, positions in tqdm(all_data):
        results, diversities = search_result(
            model, pyg_data, demands.to('cuda:0'), distances.to('cuda:0'), positions.to('cuda:0'), generate, t_diff,iter, time_limit-0.5
        )
        results_total += results
        diversities_total += diversities
        t+=1
        #print(sum_results/t,results)
    end = time.time()
    return results_total / len(all_data), diversities_total / len(all_data), end - begin


@torch.no_grad()
def search_result(model, model_data, demands, distances, positions, generate, t_dif, n_iter, time):
    model.eval()
    heu = model(model_data)
    heu_inp = model.reshape(model_data, heu) + EPS
    heu_inp=heu_inp.to('cuda:0')
    search = search_route(
        genetate=generate,
        heuristic=heu_inp,#cuda()
        demand=demands,
        distances=distances,
        device='cuda:0',
        positions=positions,
    )
    value = torch.zeros(size=(len(t_dif),))
    div = torch.zeros(size=(len(t_dif),))
    for i, t in enumerate(t_dif):
      if i==len(t_dif)-1:
        value[i], div[i] = search.test(n_iter, time)
        path = divide_routes(search.shortest_path)
        valid, length = validate(distances, demands, path)
        if valid is False:
           print("invalid solution.")
    return value, div

def validate(distance, demands, generate_r):
    val = True
    len_r = 0.0
    vis = {0}
    for r in generate_r:
        route_demand = demands[r].sum().item()
        if route_demand > 1.000001:
            val = False
        len_r += distance[r[:-1], r[1:]].sum().item()
        vis.update(r.tolist())

    if len(vis) != distance.size(0):
        val = False
    return val, len_r

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("-k", "--k_sparse", type=int, default=None, help="k_sparse")
    parser.add_argument("-p", "--path", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("-i", "--iter", type=int, default=0, help="Number of iterations to run")
    parser.add_argument("-n", "--generate", type=int, default=100, help="Number of generate")
    parser.add_argument("-d", "--device", type=str,
                        default=("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="The device to train NNs")
    parser.add_argument("-t", "--time", type=float, default=0.0, help="limit time should larger than 1")
    args = parser.parse_args()
    if args.path is not None and not os.path.isfile(args.path):
        print(f"Checkpoint file '{args.path}' not found!")
        exit(1)
    args.k_sparse = args.nodes // 5
    DEVICE = args.device if torch.cuda.is_available() else 'cpu'
    # seed everything
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    main(
        args.path,
        args.nodes,
        args.k_sparse,
        args.generate,
        args.iter,
        args.time
    )

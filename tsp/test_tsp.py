import os
import random
import time

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from net import Net
from search import search_
from utils import load_test_dataset
#python test_tsp.py 200 -p 

EPS = 1e-10
START_NODE = None

def main(chechkpoint, scale, k_sparse, size=None, generate=100, iter=10):
    test_list = load_test_dataset(scale, k_sparse, DEVICE, start_node=START_NODE)
    test_list = test_list[:(size or len(test_list))]

    t = list(range(1, iter + 1))

    if chechkpoint is not None:
        net = Net(gfn=True, Z_out_dim=1, start_node=START_NODE).to(DEVICE)
        net.load_state_dict(torch.load(chechkpoint, map_location=DEVICE))
    else:
        net = None
    avg_cost, avg_diversity, duration = validate(test_list, net, generate, t)
    print('total duration: ', duration)
    print(f"avg. cost {avg_cost[0]}, avg. diversity {avg_diversity[0]}")

@torch.no_grad()
def validate(all_data, model, generate, iter):
    begin = time.time()
    _iter = [0] + iter
    process_ = [_iter[i + 1] - _iter[i] for i in range(len(_iter) - 1)]

    value_all = torch.zeros(size=(len(process_), ))
    div_all = torch.zeros(size=(len(process_), ))
    for model_inp, distances in tqdm(all_data):
        value, div = search_result(model, model_inp, distances, generate, process_)
        value_all += value
        div_all += div
    end = time.time()
    return value_all / len(all_data), div_all / len(all_data), end - begin

@torch.no_grad()
def search_result(model, model_inp, distances, generate, iter):
    model.eval()
    her = model(model_inp)
    mat = model.reshape(model_inp, her) + EPS

    search = search_(
        distances.cpu(),
        generate,
        heuristic=mat.cpu() ,
        device='cpu',
    )
    value = torch.zeros(size=(len(iter),))
    div = torch.zeros(size=(len(iter),))
    for i, t in enumerate(iter):
        value[i], div[i] = search.val(t, inference=True, start_node=START_NODE)
    return value, div



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", type=int, help="Problem scale")
    parser.add_argument("-k", "--k_sparse", type=int, default=None, help="k_sparse")
    parser.add_argument("-p", "--path", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("-i", "--iter", type=int, default=1, help="Number of iterations to run")
    parser.add_argument("-n", "--generate", type=int, default=100, help="Number of generate")
    parser.add_argument("-d", "--device", type=str,
                        default=("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="The device to train NNs")
    parser.add_argument("-s", "--size", type=int, default=None, help="Number of instances to test")
    args = parser.parse_args()

    if args.k_sparse is None:
        args.k_sparse = args.nodes // 10

    DEVICE = args.device if torch.cuda.is_available() else 'cpu'

    # seed everything
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    if args.path is not None and not os.path.isfile(args.path):
        print(f"Checkpoint file '{args.path}' not found!")
        exit(1)

    main(
        args.path,
        args.nodes,
        args.k_sparse,
        args.size,
        args.generate,
        args.iter,
    )

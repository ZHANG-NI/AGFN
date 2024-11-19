import os
import math
import random
import time
import copy
from tqdm import tqdm
import numpy as np
import torch
from search_train1 import search1
from net import Net
from netd import Net_D
from search_train import search
from utils import backward_calculate,gen_pyg_data, load_val_dataset, gen_instance

#python train_cvrp.py 200
EPS = 1e-10
T = 5

def des_F(
        model,
        model_D,
        optimizer,
        optimizer_D,
        dataset,
        generate,
        alpha=1.0,
    ):
    model.eval()
    model_D.train()
    allloss_D = torch.tensor([0.0], device=DEVICE)
    num = 0

    for model_inp, demands, distances, positions in dataset:
        her,flow= model(model_inp, True)
        mat = model.reshape(model_inp, her) + EPS#边信息，pf,Z为流
        heu_vec_D,_= model_D(model_inp, True)
        mat_D = model_D.reshape(model_inp, heu_vec_D) + EPS#边信息，pf,Z为流
        target =mat_D
        search_route = search(
            distances=distances.to(DEVICE),
            demand=demands.to(DEVICE),
            generate=generate,
            heuristic=mat.to(DEVICE),
            heuristic_target=target.to(DEVICE),
            device=DEVICE,
            positions=positions
        )

        costs_raw, forward, paths,forward_D = search_route.generate_route(alpha)
        R=torch.exp(forward_D)
        R=R.mean(0)
        #print(R.mean())
        loss_D1=torch.pow(R,2)
        
        allloss_D+=loss_D1.mean()
        num += 1

    loss_D=allloss_D/ num
    optimizer_D.zero_grad()
    loss_D.backward()
    torch.nn.utils.clip_grad_norm_(parameters=model_D.parameters(), max_norm=3.0, norm_type=2)  # type: ignore
    optimizer_D.step()

def des_T(
        model,
        model_D,
        optimizer_D,
        dataset,
        generate,
        alpha=1.0,
    ):
    model.eval()
    model_D.train()
    allloss_D = torch.tensor([0.0], device=DEVICE)
    num = 0
    
    for model_inp, demands, distances, positions in dataset:
        her,flow= model(model_inp, True)
        mat = model.reshape(model_inp, her) + EPS
        her_D,_= model_D(model_inp, True)
        mat_D = model_D.reshape(model_inp, her_D) + EPS
        target =mat_D
        search_route = search1(
            distances=distances.to(DEVICE),
            demand=demands.to(DEVICE),
            heuristic=mat.to(DEVICE),
            heuristic_target=target.to(DEVICE),
            generate=generate,
            device=DEVICE,
            positions=positions
        )
        
        _, forward_D, paths = search_route.generate_route(alpha)
        R=torch.exp(forward_D)
        R=R.mean(0)
        #print(R)
        loss_D=torch.pow(1-R,2)
        allloss_D+=loss_D

        num += 1
    loss_D=allloss_D/ num
    optimizer_D.zero_grad()
    loss_D.backward()
    torch.nn.utils.clip_grad_norm_(parameters=model_D.parameters(), max_norm=3.0, norm_type=2)  # type: ignore
    optimizer_D.step()


def generate_train(
        model,
        model_D,
        optimizer,
        optimizer_D,
        dataset,
        generate,
        beta=1.0,
        alpha=100.0,
    ):
    model.train()
    model_D.eval()
    allloss = torch.tensor(0.0, device=DEVICE)
    num= 0

    for model_inp, demands, distances, positions in dataset:
        her,flow= model(model_inp, True)
        mat = model.reshape(model_inp, her) + EPS
        flow = flow[0]
        her_D,_= model_D(model_inp, True)
        mat_D = model_D.reshape(model_inp, her_D) + EPS
        target =mat_D
        search_route = search(
            distances=distances.to(DEVICE),
            demand=demands.to(DEVICE),
            generate=generate,
            heuristic=mat.to(DEVICE),
            heuristic_target=target.to(DEVICE),
            device=DEVICE,
            positions=positions
        )       
        costs, forward, paths,forward_D= search_route.generate_route(beta)
        cost_n = costs - costs.mean() 
        cost_n = cost_n
        R=torch.exp(forward_D)
        R=R.mean(0)
        forward = forward.sum(0) + flow.expand(generate)
        backward = backward_calculate(paths.T)-(1-R)* alpha- cost_n.detach()* alpha 
        tb = torch.pow(forward - backward, 2).mean()
        allloss += tb
        num += 1

    allloss = allloss / num
    loss = allloss
    loss = allloss / num
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=3.0, norm_type=2)  # type: ignore
    optimizer.step()


def train_data(count, n_node, k_sparse):
    for _ in range(count):
        instance = demands, distance, _ = gen_instance(n_node, DEVICE, False)
        yield gen_pyg_data(demands, distance, DEVICE, k_sparse), *instance

def validate(model, model_inp, demands, distances, positions, generate):
    model.eval()
    her = model(model_inp)
    mat = model.reshape(model_inp, her) + EPS

    search_route = search(distances=distances,demand=demands,generate=generate,heuristic=mat,device=DEVICE,positions=positions)

    costs, _, _,_ = search_route.generate_route()
    base = costs.mean().item()
    best = costs.min().item()
    best1, div1 = search_route.vali_(n_iterations=1)
    bestT, divT = search_route.vali_(n_iterations=T - 1)
    return np.array([base, best, best1, bestT, div1, divT])

@torch.no_grad()
def val(dataset, generate, net, epoch, steps_per_epoch):
    res = []
    for data, demands, distances, positions in tqdm(dataset, desc="Val"):
        res.append(validate(net, data, demands, distances, positions, generate))
    ress = [i.item() for i in np.stack(res).mean(0)]

    ##################################################
    print(f"epoch {epoch}:", ress)
    return ress[3]

def train_20step(
    node,
    k_sparse,
    generate,
    epoch,
    steps_per_epoch,
    net,
    net_D,
    optimizer,
    optimizer_D,
    batch_size=1,
    alpha=1.0,
    beta=100.0,
):
    for i in tqdm(range(steps_per_epoch), desc="Train"):
        it = (epoch - 1) * steps_per_epoch + i
        data = train_data(batch_size, node, k_sparse)
        if epoch%5!=1:
            print("1")
            generate_train(net,net_D,optimizer,optimizer_D, data, generate,  alpha, beta)
        else:
            if random.uniform(0,1)<0.7:
                print("2")
                des_F(net,net_D,optimizer,optimizer_D, data, generate, alpha)
            else:
                print("3")
                des_T(net,net_D,optimizer_D, data, 1,  alpha)


def main(
        n_nodes,
        k_sparse,
        generate,
        val_generate,
        steps,
        epochs,
        lr=7e-4,
        lr1=1e-3,
        batchs=3,
        vals=None,
        inte=5,
        path="../pretrained/cvrp_nls",
        run_name="",
        alpha_schedule_params=(0.8, 1.0, 5),  
        beta_schedule_params=(50, 500, 5),  # (beta_min, beta_max, beta_flat_epochs)
    ):
    sum_time = 0
    net = Net(gfn=True, Z_out_dim= 1).to(DEVICE)
    path = os.path.join(path, str(n_nodes), run_name)
    os.makedirs(path, exist_ok=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr * 0.1)
    net_D=Net_D(gfn=True, Z_out_dim= 1).to(DEVICE)
    optimizer_D = torch.optim.AdamW(net_D.parameters(), lr=lr1)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, epochs, eta_min=lr1 * 0.1)
    valdata = load_val_dataset(n_nodes, k_sparse, DEVICE, False)
    valdata = valdata[:(vals or len(valdata))]
    bestr = val(valdata, val_generate, net, 0, steps)

    
    for epoch in range(1, epochs + 1):
        # Beta Schedule
        start = time.time()
        beta_min, beta_max, beta_flat_epochs = beta_schedule_params
        beta = beta_min + (beta_max - beta_min) * min(math.log(epoch) / math.log(epochs - beta_flat_epochs), 1.0)

        alpha_min, alpha_max, alpha_flat_epochs = alpha_schedule_params
        alpha = alpha_min + (alpha_max - alpha_min) * min((epoch - 1) / (epochs - alpha_flat_epochs), 1.0)

        train_20step(n_nodes,k_sparse,generate,epoch,steps,net,net_D,optimizer,optimizer_D,batchs,alpha,beta,)
        sum_time += time.time() - start

        if epoch % inte == 0:
            curr_result = val(valdata, val_generate, net, epoch, steps)
            if curr_result < bestr:
                torch.save(net.state_dict(), os.path.join(path, f'best.pt'))
                bestr = curr_result
            torch.save(net.state_dict(), os.path.join(path, f"{epoch}.pt"))

        scheduler.step()
        scheduler_D.step()

    print('\ntotal training duration:', sum_time)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", metavar='N', type=int, help="Problem scale")
    parser.add_argument("-k", "--k_sparse", type=int, default=None, help="k_sparse")
    parser.add_argument("-l", "--lr", metavar='η', type=float, default=5e-4, help="Learning rate")
    parser.add_argument("-d", "--device", type=str, 
                        default=("cuda:0" if torch.cuda.is_available() else "cpu"), 
                        help="The device to train NNs")
    parser.add_argument("-a", "--generate", type=int, default=20, help="Number of generate")
    parser.add_argument("-va", "--vgenerate", type=int, default=20, help="Number of generate for validation")
    parser.add_argument("-b", "--batchs", type=int, default=10, help="Batch size")#10
    parser.add_argument("-s", "--steps", type=int, default=20, help="Steps per epoch")#20
    parser.add_argument("-e", "--epochs", type=int, default=5000, help="Epochs to run")
    parser.add_argument("-v", "--vals", type=int, default=5, help="Number of instances for validation")#5
    parser.add_argument("-o", "--path", type=str, default="../pretrained/cvrp_nls",
                        help="The directory to store checkpoints")
    parser.add_argument("--inte", type=int, default=1, help="The interval to validate model")

    parser.add_argument("--run_name", type=str, default="", help="Run name")

    parser.add_argument("--alpha_min", type=float, default=0.8, help='alpha')
    parser.add_argument("--alpha_max", type=float, default=1.0, help='alpha')
    parser.add_argument("--alpha_flat_epochs", type=int, default=5, help='alpha flat epochs')

    parser.add_argument("--beta_min", type=float, default=None, help='Beta min')
    parser.add_argument("--beta_max", type=float, default=None, help='Beta max ')
    parser.add_argument("--beta_flat_epochs", type=int, default=5, help='Beta flat epochs')


    args = parser.parse_args()

    args.k_sparse = args.nodes // 5

    beta_min_map = {100: 200, 200: 500, 400: 500, 500: 500, 1000: 2000}
    args.beta_min = beta_min_map[args.nodes]
    beta_max_map = {100: 1000, 200: 2000, 400: 2000, 500: 2000, 1000: 2000}
    args.beta_max = beta_max_map[args.nodes]

    DEVICE = args.device if torch.cuda.is_available() else "cpu"
    # seed everything
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    run_name = f"[{args.run_name}]" if args.run_name else ""

    main(
        args.nodes,
        args.k_sparse,
        args.generate,
        args.vgenerate,
        args.steps,
        args.epochs,
        #lr=args.lr,
        batchs=args.batchs,
        vals=args.vals,
        inte=args.inte,
        path=args.path,
        run_name=run_name,
        alpha_schedule_params=(args.alpha_min, args.alpha_max, args.alpha_flat_epochs),
        beta_schedule_params=(args.beta_min, args.beta_max, args.beta_flat_epochs),
    )#False#(not args.disable_guided_exp),

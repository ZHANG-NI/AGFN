import numpy as np
import subprocess
import torch
from tqdm import tqdm
import pyautogui
#from torch_geometric.data import Data
import re
import os

def write_instance(instance, instance_name, instance_filename):

    distance_matrix, demands, capacity = instance
    n_nodes = len(demands)
    with open(instance_filename, "w") as f:
        f.write("NAME : " + instance_name + "\n")
        f.write("COMMENT : blank\n")
        f.write("TYPE : CVRP\n")
        f.write("DIMENSION : " + str(n_nodes) + "\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        f.write("CAPACITY : " + str(capacity) + "\n")
        f.write("EDGE_WEIGHT_SECTION\n")

        for row in distance_matrix:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("DEMAND_SECTION\n")
        for i in range(n_nodes):
            f.write(str(i + 1) + " " + str(demands[i]) + "\n")

        f.write("DEPOT_SECTION\n 1\n -1\n")
        f.write("EOF\n")

def write_candidate(n,dataset_name, instance_name, distance_matrix, heu_vec, filename="example_candidate.txt"):

    n_nodes = len(distance_matrix)
    candidate = []

    for i in range(n_nodes):
        #print(heu_vec)
        edges = [(j, heu_vec[i, j].item()) for j in range(n_nodes) if i != j]
        edges.sort(key=lambda x: x[1], reverse=True)#
        t=n_nodes//5-2
        candidate.append(edges[:t])  
    with open(filename, "w") as f:
        f.write(f"{n}\n")
        for i in range(n):
          if i<n_nodes:
            line = f"{i + 1} 0 {len(candidate[i])}"
            for j in range(len(candidate[i])):
                line += f" {candidate[i][j][0] + 1} {int(1/candidate[i][j][1])-int(1/candidate[i][0][1])}"
          else:
            line = f"{i + 1} 0 {len(candidate[0])}"
            for j in range(len(candidate[0])):
                line += f" {candidate[0][j][0] + 1} {int(1/candidate[0][j][1])-int(1/candidate[0][0][1])}"
          f.write(line + "\n")
        f.write("-1\nEOF\n")

def write_para(dataset_name, instance_name, instance_filename, method, para_filename, max_trials=1000, seed=1234):

    with open(para_filename, "w") as f:
        f.write("PROBLEM_FILE = " + instance_filename + "\n")
        f.write("SEED = " + str(seed) + "\n")
        if method == "NeuroLKH":
            f.write("CANDIDATE_SET_TYPE = NEAREST-NEIGHBOR\n")
            f.write("MTSP_SOLUTION_FILE= = out_new_1.txt\n")
            f.write("SPECIAL\nRUNS = 1\n")
            f.write("MAX_TRIALS = " + str(max_trials) + "\n")
            f.write("SUBGRADIENT = NO\n")
            f.write("MAX_SWAPS = 8 \n")
            f.write("MAX_CANDIDATES = 5 \n")
            f.write("CANDIDATE_FILE = " + instance_name + ".txt\n")
            f.write("TOUR_FILE = mid.txt\n")
        elif method == "NeuroLKH1":
            f.write("MTSP_SOLUTION_FILE= = out_new_2.txt\n")
            f.write("INITIAL_TOUR_FILE = mid.txt\n")
            f.write("MAX_TRIALS = " + str(max_trials) + "\n")
            f.write("SPECIAL\nRUNS = 1\n")
            f.write("SUBGRADIENT = NO\n")
        elif method == "FeatGenerate":
            f.write("MTSP_SOLUTION_FILE= = out.txt\n")
            f.write("MAX_TRIALS = " + str(max_trials) + "\n")
            f.write("SPECIAL\nRUNS = 1\n")
            f.write("SUBGRADIENT = NO\n")
            f.write("CANDIDATE_FILE = " + instance_name + ".txt\n")
        else:
            assert method == "LKH"
def generate_lkh(heu_mat,demand,distance):
  demands = demand*100


  capacity = 100

  distance_matrix = distance*100

  instance = (distance_matrix.int().tolist(), demands.int().tolist(), capacity)
  instance_name = "example_cvrp"
  instance_filename = "example_instance.cvrp"
  para_filename = "example_instance.par"
  candidate_filename = "example_candidate.txt"

  write_instance(instance, instance_name, instance_filename)
  write_para("example_dataset", "example_instance", instance_filename, "FeatGenerate", para_filename,20)
  import subprocess
  with open('lkh_log.txt', 'w') as log_file:
    res = subprocess.Popen(
              ["LKH-3.exe", para_filename],
              #check=True,
              stdout=subprocess.PIPE,
              stderr=subprocess.PIPE,
              text=True
          )
    while True:
                output = res.stdout.readline()
                if output == '' and res.poll() is not None:
                    break
                if output:
                    log_file.write(output)

                    if "Best CVRP solution" in output:
                        res.terminate()

  filename = 'example_instance.txt'

  with open(filename, 'r') as file:
      lines = file.readlines()
  os.remove("example_instance.txt")

  with open('out.txt', 'r') as file:
        text = file.read()

  pattern = re.compile(r'1(?:\s\d+)*\s1')

  paths = pattern.findall(text)

  path= [int(node)-1 for path in paths for node in path.split()[0:-1] ]+ [0]


  return torch.tensor(path).view(-1, 1).to("cuda:0")
# ADVERSARIAL GENERATIVE FLOWNETWORK FOR SOLVING VEHICLE ROUTING PROBLEMS

See 

[Adversarial Generative Flow Network for Solving Vehicle Routing Problems]: https://openreview.net/forum?id=tBom4xOW1H

 for the paper associated with this codebase.

## CVRP

##### Training

```cmd
$cd cvrp
$python train_cvrp.py $N
```

$N is the nodes of the instances

##### Testing

```cmd
$cd cvrp
$python test_cvrp.py $N -p "path_to_checkpoint"
```

## TSP

##### Training

```cmd
$cd tsp
$python train_cvrp.py $N
```

$N is the nodes of the instances

##### Testing

```cmd
$cd tsp
$python test_cvrp.py $N -p "path_to_checkpoint"
```

## Reference

If you find this codebase useful, please consider citing the paper:

```
@inproceedings{zhangadversarial,
  title={Adversarial Generative Flow Network for Solving Vehicle Routing Problems},
  author={Zhang, Ni and Yang, Jingfeng and Cao, Zhiguang and Chi, Xu},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```


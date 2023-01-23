# [PRP-NAS] Pareto Rank-preserving SuperNetwork for HW-NAS
This repository provides details and implementation associated with the paper "Pareto Rank-preserving SuperNetwork for HW-NAS". 
We describe in this repository a methodology to train a supernetwork to accurately rank the architectures in a multi-objective context. 

* Our supernetworks achieve a ~97% near optimal Pareto front on common NAS Benchmarks, including DARTS, NAS-Bench-201 and ProxylessNAS.
* The methodology is tested on several hardware platforms, including Jetson Nano, Pixel 3, Raspberry Pi3 and FPGA ZCU102. We further show a way to deploy a small supernetwork (corresponding to the final Pareto approximation) to enhance battery life and energy consumption 
* Our supernetwork weights are available in this link. 

## How to use

* **On a novel dataset:**
    1. Add a *data* folder with your new dataset 
    2. Execute the script *build_dataset.sh* which will generate specific batches with Pareto fronts. (This may take few minutes)
    3. run *train.py* with a selected *architecture* from the search spaces and a specific number of epochs. 

``` python train.py --arch 201 --epochs 10 --data ./data ```

* **On CIFAR-10/ImageNet:** 
CIFAR-10 is selected by default. For ImageNet just use *--data imagenet*. 

``` python train.py --arch 201 --epochs 10```

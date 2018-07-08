# Deep Reasoning with Knowledge Graph for Social Relationship Understanding

This repo includes the source code of the paper: "[Deep Reasoning with Knowledge Graph for Social Relationship Understanding](https://arxiv.org/abs/1807.00504)" (IJCAI 2018) by Zhouxia Wang, Tianshui Chen, Jimmy Ren, Weihao Yu, Hui Cheng, Liang Lin.

## Environment

The code is implemented using the Pytorch library with Python 2.7 and has been tested on a desktop with the system of Ubuntu 14.04 LTS.

## Dataset
[PISC](https://zenodo.org/record/1059155#.WznPu_F97CI) was released by [[Li et al. ICCV 2017](https://arxiv.org/abs/1708.00634)]. It involves two-level relationship, i.e., coarse-level relationships with 3 categories and fine-level relationships with 6 categories.

[PIPA-relation](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/human-activity-recognition/social-relation-recognition/) was released by [[Sun et al. CVPR 2017](https://arxiv.org/abs/1704.06456)]. It covers 5 social domains, which can be further divided into 16 social relationships. On this dataset, we focus on the 16 social relationships.

## Models && objects && adjacency matrices
Models, objects and ajacency matrices are in [HERE](https://pan.baidu.com/s/13tvWT5FmfvIFaBRE9nq1WQ).

## Usage
    usage: test.py [-h] [-j N] [-b N] [--print-freq N] [--weights PATH]
               [--scale-size SCALE_SIZE] [--world-size WORLD_SIZE] [-n N]
               [--write-out] [--adjacency-matrix PATH] [--crop-size CROP_SIZE]
               [--result-path PATH]
               DIR DIR DIR

    PyTorch Relationship

    positional arguments:
      DIR                       path to dataset
      DIR                       path to objects (bboxes and categories of objects)
      DIR                       path to test list

    optional arguments:
      -h, --help                show this help message and exit
      -j N, --workers N         number of data loading workers (defult: 4)
      -b N, --batch-size N      mini-batch size (default: 1)
      --print-freq N, -p N      print frequency (default: 10)
      --weights PATH            path to weights (default: none)
      --scale-size SCALE_SIZE   input size
      --world-size WORLD_SIZE   number of distributed processes
      -n N, --num-classes N     number of classes / categories
      --write-out               write scores
      --adjacency-matrix PATH   path to adjacency-matrix of graph
      --crop-size CROP_SIZE     crop size
      --result-path PATH        path for saving result (default: none)

## Test
Modify the path of data before running the script.

    sh test.sh
    
## Result

PISC: Coarse-level

Methods|Intimate|Non-Intimate|No Relation|mAP
-|-|-|-|-
Union CNN  | 72.1 | 81.8 | 19.2| 58.4
Pair CNN  | 70.3 | 80.5 | 38.8 | 65.1
Pair CNN + BBox + Union  | 71.1 | 81.2 | 57.9 | 72.2
Pair CNN + BBox + Global | 70.5 | 80.0 | 53.7 | 70.5
Dual-glance | 73.1 | **84.2** | 59.6 | 79.7 | 35.4 | 79.7
Ours | **81.7** | 73.4 | **65.5** | **82.8**

PISC: Fine-level

Methods|Friends|Family|Couple|Professional|Commercial|No Relation|mAP
-|-|-|-|-|-|-|-
Union CNN | 29.9 | 58.5 | 70.7 | 55.4 | 43.0 | 19.6 | 43.5
Pair CNN  | 30.2 | 59.1 | 69.4 | 57.5 | 41.9 | 34.2 | 48.2
Pair CNN + BBox + Union  | 32.5 | 62.1 | 73.9 | 61.4 | 46.0 | 52.1 | 56.9
Pair CNN + BBox + Global | 32.2 | 61.7 | 72.6 | 60.8 | 44.3 | 51.0 | 54.6
Dual-glance | 35.4 | **68.1** | **76.3** | 70.3 | **57.6** | 60.9 | 63.2
Ours | **59.6** | 64.4 | 58.6 | **76.6** | 39.5 | **67.7** | **68.7**

PIPA-relation: 

Methods   | accuracy 
-|-
Two stream CNN | 57.2
Dual-Glance | 59.6 
Ours  | **62.3**

## Citation
    @inproceedings{Wang2018Deep,
        title={Deep Reasoning with Knowledge Graph for Social Relationship Understanding},
        author={Zhouxia Wang, Tianshui Chen, Jimmy Ren, Weihao Yu, Hui Cheng, Liang Lin},
        booktitle={International Joint Conference on Artificial Intelligence},
        year={2018}
    }

## Contributing
For any questions, feel free to open an issue or contact us (zhouzi1212@gmail.com & tianshuichen@gmail.com)

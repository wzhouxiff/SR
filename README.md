# Deep Reasoning with Knowledge Graph for Social Relationship Understanding

This repo includes the source code of the paper: "Deep Reasoning with Knowledge Graph for Social Relationship Understanding" (IJCAI 2018) by Zhouxia Wang, Tianshui Chen, Jimmy Ren, Weihao Yu, Hui Cheng, Liang Lin.

## Environment

The code is tested on 64 bit Linux (Ubuntu 14.04 LTS), and besed on Pytorch with Python 2.7.

## Dataset
[PISC](https://zenodo.org/record/1059155#.WznPu_F97CI) was released by J. Li et al. in ICCV 2017. It involves two-level relationship, coarse-level relationship(alias domain) which has 3 categories and fine-level relationship which has 6 categories. More detail can be found in the [link](https://zenodo.org/record/1059155#.WznPu_F97CI) or in our paper.

[PIPA-relation](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/human-activity-recognition/social-relation-recognition/) was released by Q. Sun et al. in CVPR 2017. It alse involves two-level relationship, 5 social domains and 16 social relations. In our experiment, we only focus on the 16 social relations.

## Model && object box && adjacency matrix
Will be released soon...

## Test

    sh test.sh
    
## Result

PISC - Coarse-level

Methods|Intimate|Non-Intimate|No Relation|mAP
-|-|-|-|-
Union CNN  | 72.1 | 81.8 | 19.2| 58.4
Pair CNN  | 70.3 | 80.5 | 38.8 | 65.1
Pair CNN + BBox + Union  | 71.1 | 81.2 | 57.9 | 72.2
Pair CNN + BBox + Global | 70.5 | 80.0 | 53.7 | 70.5
Dual-glance | 73.1 | **84.2** | 59.6 | 79.7 | 35.4 | 79.7
Ours | **81.7** | 73.4 | **65.5** | **82.8**

PISC - Fine-level

Methods|Friends|Family|Couple|Professional|Commercial|No Relation|mAP
-|-|-|-|-|-|-|-
Union CNN | 29.9 | 58.5 | 70.7 | 55.4 | 43.0 | 19.6 | 43.5
Pair CNN  | 30.2 | 59.1 | 69.4 | 57.5 | 41.9 | 34.2 | 48.2
Pair CNN + BBox + Union  | 32.5 | 62.1 | 73.9 | 61.4 | 46.0 | 52.1 | 56.9
Pair CNN + BBox + Global | 32.2 | 61.7 | 72.6 | 60.8 | 44.3 | 51.0 | 54.6
Dual-glance | 35.4 | **68.1** | **76.3** | 70.3 | **57.6** | 60.9 | 63.2
Ours | **59.6** | 64.4 | 58.6 | **76.6** | 39.5 | **67.7** | **68.7**
## Citation
    

## Contributing
For any questions, bug(even typos) and/or features requests do not hesitate to contact me or open an issue!(zhouzi1212@gmail.com)

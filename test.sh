#!/bin/bash

# Path to Images
ImagePath=""
# Path to object boxes
ObjectsPath="Path/SR/objects/PISC_objects/"
# Path to test list
TestList="Path/SR/list/PISC_fine_level_test.txt"
# Path to adjacency matrix
AdjMatrix="Path/SR/adjacencyMatrix/PISC_fine_level_matrix.npy"
# Number of classes
num=6
# Path to save scores
ResultPath=""
# Path to model
ModelPath="Path/SR/models/PISC_fine_level.pth.tar"
echo $ModelPath

python test.py $ImagePath $ObjectssPath $TestList --weight $ModelPath --adjacency-matrix $AdjMatrix -n $num -b 1 --print-freq 100 --write-out --result-path $ResultPath


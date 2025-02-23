# SA-Seg

Code for: SA-Seg: Annotation-Efficient Segmentation for Airway Tree Using Saliency-based Annotation.

# Usage 
training with
```
python3 main.py 
--gpu 0  
--fp16  
--ds BAS # dataset
--label_type part 
--part_type pu_label 
--mlp branch
--warm_up 20
--lambda_ 1.0 
--mu_ 0.5 
--loss gul
```

# Requirements 

This code has been tested with
```
Python 3.7
torch 1.11
batchgenerators 0.24 
SimpleITK 2.1.1.2
numpy 1.21.5
pandas 1.3.5
scikit-image 0.19.2
scipy 1.7.3
```

# MAE-3D

This is the source code for our proposed __Masked Autoencoders in 3D Point Cloud Representation Learning (MAE3D)__. 

## Requirements
python >= 3.7
pytorch >= 1.7.0
numpy
scikit-learn
einops
h5py
tqdm
and if you are first time to run "pointnet2_ops_lib", you need

```
pip install pointnet2_ops_lib/.
```

## Datasets
The main datasets we used in our project are ShapeNet and ModelNet40, and you can download them in:
[ShapeNet](https://livebournemouthac-my.sharepoint.com/:u:/g/personal/jiangj_bournemouth_ac_uk/EccpUcO2xV1Or99sQ5-eVpIByCmHibWCaYrlPKsVujkP_g?e=ELwsuY) and [ModelNet40](https://livebournemouthac-my.sharepoint.com/:f:/g/personal/jiangj_bournemouth_ac_uk/EqLWT4NyRDNPn-mRrcexbhkB5b3woe1Hiu5jVHrQXEQmjg?e=XmrBPw).

and then you need to put them at "./data"

## Evaluate
We can get an accuracy of 93.4% (SOTA) on the ModelNet40.
The pre-trained model can be found in [here](https://livebournemouthac-my.sharepoint.com/:u:/g/personal/jiangj_bournemouth_ac_uk/EfvO9tJcHylOv6x72UeffnEBacsbT6YNchA1ruF1iQP1Dg?e=5EuTSx).

You need to move the "model_cls.t7" to "./checkpoints/mask_ratio_0.7/exp_shapenet55_block/models", 
then you can simply restore our model and evaluate on ModelNet40 by

```
python main_cls.py --exp_name exp_shapenet55_block --mask_ratio 0.7 --eval True
```

##  Pre-training: Point Cloud Completion
You should download ShapeNet dataset first, and then simply run
```
python main_pretrain.py --exp_name exp_shapenet55_block --mask_ratio 0.7
```
If you want to visualize all reconstructed point cloud (will spend a lot of time), you could run as
```
python main_pretrain.py --exp_name exp_shapenet55_block --mask_ratio 0.7 --visualize True
```

##  Fine-tuning: supervised classification
You should download ModelNet dataset first, and then simply run
```
python main_cls.py --exp_name exp_shapenet55_block --mask_ratio 0.7 --pretrained True --finetune True
```

##  Linear classifier: unsupervised classification
You should download ModelNet dataset first, and then simply run
```
python main_cls.py --exp_name exp_shapenet55_block --mask_ratio 0.7 --pretrained True --linear_classifier True 
```

## Citation
If you find our work useful, please consider citing:
```
@article{jiang2023masked,
    title={Masked autoencoders in 3d point cloud representation learning},
    author={Jiang, Jincen and Lu, Xuequan and Zhao, Lizhi and Dazaley, Richard and Wang, Meili},
    journal={IEEE Transactions on Multimedia},
    year={2023},
    publisher={IEEE}
}    
```

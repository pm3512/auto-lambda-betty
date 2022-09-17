# Auto-Lambda (Betty)
This repository contains the implementation of [AutoLambda](https://github.com/lorenmt/auto-lambda) using [Betty](https://github.com/leopard-ai/betty).

## Running
Training Auto-Lambda in Multi-task / Auxiliary Learning Mode:
```
python trainer_dense_betty.py --dataset [nyuv2, cityscapes] --task [PRIMARY_TASK] --weight autol --gpu 0   # for NYUv2 or CityScapes dataset
python trainer_cifar_betty.py --subset_id [PRIMARY_DOMAIN_ID] --weight autol --gpu 0   # for CIFAR-100 dataset
```



## Citation
If you found this code/work to be useful in your own research, please considering citing AutoLambda and Betty:

```
@article{liu2022auto_lambda,
    title={Auto-Lambda: Disentangling Dynamic Task Relationships},
    author={Liu, Shikun and James, Stephen and Davison, Andrew J and Johns, Edward},
    journal={Transactions on Machine Learning Research},
    year={2022}
}

@article{choe2022betty,
  title={Betty: An Automatic Differentiation Library for Multilevel Optimization},
  author={Choe, Sang Keun and Neiswanger, Willie and Xie, Pengtao and Xing, Eric},
  journal={arXiv preprint arXiv:2207.02849},
  year={2022}
}
```
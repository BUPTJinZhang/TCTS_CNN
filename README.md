# TCTS_CNN
Code for the paper “Temporal Consistency Two-Stream CNN for Human Motion Prediction”<br>

### Dependencies

* cuda 9.0
* Python 3.6
* Tensorflow 1.8.0.

### Training commands
To train on 3D space,
```bash
bash scripts/h36m/Short_term_train.py
```

### Results

* Human3.6-short-term prediction on 3D coordinate

|                | 80ms   | 160ms  | 320ms  | 400ms  |
|----------------|------|------|------|------|
| MPJPE | 9.8 | 22.6 | 48.1 | 58.4 |

### Citing

If you use our code, please cite our work

```
@article{Tang2021TemporalCT,
  title={Temporal Consistency Two-Stream CNN for Human Motion Prediction},
  author={Jin Tang and Jin Zhang and Jianqin Yin},
  journal={ArXiv},
  year={2021},
  volume={abs/2104.05015}
}
```

### Acknowledgments

Some of our evaluation code and data process code was adapted/ported from [Residual Sup. RNN](https://github.com/una-dinosauria/human-motion-prediction) by [Julieta](https://github.com/una-dinosauria). The overall code framework (dataloading, training, testing etc.) is adapted from [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline). 

### Licence
BUPT



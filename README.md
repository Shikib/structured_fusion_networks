# Structured Fusion Networks

Structured Fusion Networks for Dialog https://arxiv.org/abs/1907.10016

Code written by Shikib Mehri and Tejas Srinivasan.

## Pretrained Models

The main models trained in the paper have been provided and can be used as follows:

**Sequence-to-Sequence with Attention**: 

`python3 test.py --model_name models/attn --use_attn --test_only`

**Structured Fusion Networks**: 

`python3 test.py --model_name models/sfn --structured_fusion --test_only`

**Structured Fusion Networks with Reinforcement Learning**: 

`python3 test.py --model_name models/sfn_rl --structured_fusion --test_only`

## Training

Structured Fusion Networks can be trained with the following command:

`python3 train.py --model_name 'models/sfn' --structured_fusion --use_cuda True --tune_params True`

SFNs can be fine-tuned with RL as follows (you may need to edit the code a bit):

`python3 rl_train.py --structured_fusion --model_name models/rl_sfn --lr 0.00001`

## Questions

Please feel free to send any questions/concerns to amehri@andrew.cmu.edu.

## Citing

The following paper should be cited if you use this code.

```
@article{mehri2019structured,
  title={Structured Fusion Networks for Dialog},
  author={Mehri, Shikib and Srinivasan, Tejas and Eskenazi, Maxine},
  journal={arXiv preprint arXiv:1907.10016},
  year={2019}
}
```

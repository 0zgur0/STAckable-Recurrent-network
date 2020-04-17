# STAR Network
Tensorflow  implementation for our RNN model based on our Arxiv paper:

Turkoglu, Mehmet Ozgur, et al. "Gating Revisited: Deep Multi-layer RNNs That Can Be Trained." arXiv preprint arXiv:1911.11033 (2019).

# [[Paper]](https://arxiv.org/abs/1911.11033)


<img src="https://raw.githubusercontent.com/0zgur0/STAR_Network/master/imgs/cells.PNG" width="600" height="360">
<img src="https://raw.githubusercontent.com/0zgur0/STAR_Network/master/imgs/img_gras.PNG" width="600" height="360">

## Getting Started

Run the model with 
```bash
python main.py --cell bn-star --stack 4 --data pmnist
```

## Citation
```bash
@article{turkoglu2019gating,
  title={Gating Revisited: Deep Multi-layer RNNs That Can Be Trained},
  author={Turkoglu, Mehmet Ozgur and D'Aronco, Stefano and Wegner, Jan Dirk and Schindler, Konrad},
  journal={arXiv preprint arXiv:1911.11033},
  year={2019}
}
```

The code is mostly based on the following implementation: https://github.com/JosvanderWesthuizen/janet

# STAR Network
Tensorflow  implementation for STAckable Recurrent (STAR) network based on our TPAMI paper:

Gating Revisited: Deep Multi-layer RNNs That Can Be Trained.

# [[Paper]](https://arxiv.org/abs/1911.11033)  - [[Blog post]](https://medium.com/p/2f01acdb73a7)


<img src="https://raw.githubusercontent.com/0zgur0/STAR_Network/master/imgs/cells.PNG" width="960" height="210">
<img src="https://raw.githubusercontent.com/0zgur0/STAR_Network/master/imgs/img_grad.PNG" width="960" height="540">

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

The code is mostly based on https://github.com/JosvanderWesthuizen/janet.


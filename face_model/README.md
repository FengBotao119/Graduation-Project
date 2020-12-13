## Introduction

I'm going to use [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) to extract [AUs](https://www.cs.cmu.edu/~face/facs.htm) to train a model which can capture people's expression.

## Model

- NN + Focal loss
- NN + Cross entropy loss
- Wide&Deep model + Focal loss
- Wide&Deep model + Cross entropy loss

## Dataset

- [Expression in-the-Wild (ExpW) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html)
- [Real-world Affective Faces Database](http://www.whdeng.cn/raf/model1.html)

## Results

![loss](https://github.com/FengBotao119/Graduation-Project/blob/master/face_model/results/loss.png)

![confusion_matrix](https://github.com/FengBotao119/Graduation-Project/blob/master/face_model/results/confusion_matrix.png)


## Reference

- Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]//Proceedings of the 1st workshop on deep learning for recommender systems. 2016: 7-10.
- Lin T Y, Goyal P, Girshick R, et al. Focal loss for dense object detection[C]//Proceedings of the IEEE international conference on computer vision. 2017: 2980-2988.

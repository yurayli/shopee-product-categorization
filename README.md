
# Shopee Code League - Product Categorization
This is the 7th place solution to the competition [Shopee Code League image competition](https://www.kaggle.com/c/shopee-product-detection-open) (with the dataset therein).

## Overview
<img src="imgs/img1.png" alt="drawing" width="900"/>

### TPU/XLA data pipeline
Based on the previous experience in [Kaggle flower competition](https://www.kaggle.com/c/flower-classification-with-tpus/), we decided to use the power of tpu/xla engine for training neural nets in this competition. The tools for data preprocessing is a bit different from training in normal gpu, but luckily we can still use tf.data api.
* To train lots of images, it is better firstly translate images to the format _filename.tfrecords_.
* We can use normal preprocessing package such as _albumentations_, but we need to implement the functions of image transformations by ourselves. Here we use image shift/rotate/scale and _Cutout_ for image augmentations.
* For a ResNet50, ~990 s/epoch on tesla p100; ~70 s/epoch on tpu v3-8.

### Model
We start from ResNet then change to EfficientNet as our model architecture. Also, the proper image scale is important to the corresponding network.

### Noisy labels
<img src="imgs/img2.png" alt="drawing" width="700"/><br><br>
After checking the data and the results of the baseline model, we found out the labels are noisy, which may be the reason that we cannot get high accuracy in this dataset. There is certain ambiguity between some categories, and some images are hard to categorize even if checked by our own eyes. Therefore we adopt _label smoothing_ here. The better pretrained architecture can also be more robust to noisy labels. [[1](https://ai.googleblog.com/2020/08/understanding-deep-learning-on.html)]

### Training
<img src="imgs/img3.png" alt="drawing" width="900"/><br>
The warmup-annealing learning rate schedule helps the network training more stable and converge faster [[2](https://arxiv.org/abs/1708.07120)]. We further extended the annealing part to cyclic annealing, which gives the network higher chances to find better optima. Also we can use the optima checkpoints during training for final ensembling. [[3](https://arxiv.org/abs/1704.00109), [4](https://arxiv.org/abs/1803.05407)]

### Dependencies
tensorflow 2.2.0
tensorflow-addons 0.9.1
image-classifiers 1.0.0
efficientnet 1.1.0

### References
[1] Understanding Deep Learning on Controlled Noisy Labels ([link](https://ai.googleblog.com/2020/08/understanding-deep-learning-on.html))
[2] Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates ([link](https://arxiv.org/abs/1708.07120))
[3] Snapshot Ensembles: Train 1, get M for free ([link](https://arxiv.org/abs/1704.00109))
[4] Averaging Weights Leads to Wider Optima and Better Generalization ([link](https://arxiv.org/abs/1803.05407))

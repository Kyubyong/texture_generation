# An Implementation of 'Texture Synthesis Using Convolutional Neural Networks'
This project is desgined to synthesize 28 texture classes of the [Kylberg Texture Dataset](http://www.cb.uu.se/~gustaf/texture/), based on the ideas of Gatys et al.'s paper [Texture Synthesis Using Convolutional Neural Networks](https://arxiv.org/pdf/1505.07376v3.pdf). 

## Requirements
  * numpy >= 1.11.1
  * sugartensor == 0.0.1.8 (pip install sugartensor)
  * Kylberg Texture Dataset (Can be freely downloaded [here](http://www.cb.uu.se/~gustaf/texture/data/without-rotations-zip/)
	
## Research Question
Can we generate images that are similar to the real texture images using neural networks?

## Main Idea
If two images are similar, their feature maps should be similar, and vice versa. Accordingly, first, we train discriminative networks such that they can correctly classify different classes of texture images. Then, we train generative networks such that the feature maps of the input image become similar with those of its target true image.

## Dataset
Refer to [Kylberg Texture Dataset](http://www.cb.uu.se/~gustaf/texture/)

## Model Architecture and Objective Function

Model: VGG-19, replacing the original final dense layers with a convolutional layer.<br/>
Objective function: Sum of L2 losses between the gram matrix of the feature maps of the noise and the target image.

## Folder and file instructions
  * prepro.py: Preprocessing. Download / unzip the dataset and write its path to the `Hyperaparams.image_fpath`. This file should make queues of images.
  * train.py: Training. This should train the discriminative model so that the networks can correctly classify images. By default, it creates log files and model files at `asset/train/log` and `asset/train/ckpt` respectively.
  * test.py: Testing. This should read the latest model file and print out classification results.
  * model-001-89612: Pretrained model parameters. Can be downloaded [here](https://drive.google.com/open?id=0B5M-ed49qMsDLU9SV3A3VmczV0E).
    (If you use this, copy it to `asset/train/ckpt`.)
  * gen.py: Generating. This should generate an image for the given target image to `gen_images` folder. Put the path of the target image as an argument.
	    We generated 28 images, targeting the first image of each class. Here is the simple bash script.

```
#!/bin/bash
for entry in ../datasets/Kylberg\ Texture\ Dataset\ v.\ 1.0/without-rotations-zip/*/*-a-p001.png
do
  python gen.py "$entry"
done
```

## Results

Classification acc. = 4285/4480 = 0.96<br/>

Here are the generated images.

![collection_0](image_collection/collection_0.png?raw=true)
![collection_1](image_collection/collection_1.png?raw=true)
![collection_2](image_collection/collection_2.png?raw=true)
![collection_3](image_collection/collection_3.png?raw=true)





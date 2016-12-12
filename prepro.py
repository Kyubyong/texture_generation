#/usr/bin/python2
# coding: utf-8

import numpy as np
import cPickle as pickle
import codecs
import re
import sugartensor as tf
import random

# image file path
class Hyperparams:
    image_fpath = '../../datasets/Kylberg Texture Dataset v. 1.0/without-rotations-zip/*/*.png'

class Data:
    def __init__(self, batch_size=16):
        
        print "# Make classes"
        import glob
        files = glob.glob(Hyperparams.image_fpath)
        labels = [f.split('/')[-1].split('-')[0] for f in files] # ['scarf2', 'scarf1', ...]

        self.idx2label = {idx:label for idx, label in enumerate(set(labels))}
        self.label2idx = {label:idx for idx, label in self.idx2label.items()}    
        
        labels = [self.label2idx[label] for label in labels] # [3, 4, 6, ...]
        
        files = tf.convert_to_tensor(files) #(4480,) (4480,)
        labels = tf.convert_to_tensor(labels) #(4480,) (4480,)
        
        file_q, label_q = tf.train.slice_input_producer([files, labels], num_epochs=1) #  (), ()
        img_q = tf.image.decode_png(tf.read_file(file_q), channels=1) # (576, 576, 1) uint8
        img_q = self.transform_image(img_q) # (224, 224, 1) float32
        
        self.x, self.y = tf.train.shuffle_batch([img_q, label_q], batch_size,
                                             num_threads=32, capacity=batch_size*128,
                                             min_after_dequeue=batch_size*32, 
                                             allow_smaller_final_batch=False)      # (16, 224, 224, 1) (16,)   

    def transform_image(self, img):
        r"""
        Arg:
          img: A 3-D tensor.
          
        Returns:
          A `Tensor`. Has the shape of (224, 224) and dtype of float32.
        """
        # center crop
        offset_height = (576-224)/2
        offset_width = offset_height
        img = tf.image.crop_to_bounding_box(img, offset_height, offset_width, 224, 224)
        # normalization
        img = img.sg_float() / 255
    
        return img


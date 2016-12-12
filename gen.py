# -*- coding: utf-8 -*-
'''
Texture image generation
'''
import sugartensor as tf
import numpy as np
from scipy import misc
import glob
import os, sys
import time
from prepro import Hyperparams
import sys

sample_image = sys.argv[1]

def transform_image(target_img):
    r"""
    Arg:
      target_img: img file full path
    
    Returns:
      A numpy array of (1, 224, 224, 1)
    """
    img = misc.imread(target_img)

    # Center crop
    offset_height = (576-224)/2
    offset_width = offset_height
    img = img[offset_height:offset_height+224, offset_width:offset_width+224]
    
    # Convert to 4-D
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, -1)
    
    # Normalize
    img = img.astype(np.float32) / 255

    return img

class ModelGraph:
    def __init__(self):
        
        with tf.sg_context(name='generator'):
            self.x = tf.sg_initializer.he_uniform(name="x", shape=[1, 224, 224, 1]) # noise image
            self.y = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 1]) # true target image
        
        with tf.sg_context(name='conv', act='relu'):
            self.x_conv1 = (self.x
                    .sg_conv(dim=64)
                    .sg_conv()
                    .sg_pool()) # (1, 112, 112, 64)
            self.x_conv2 = (self.x_conv1
                    .sg_conv(dim=128)
                    .sg_conv()
                    .sg_pool()) # (1, 56, 56, 128)
            self.x_conv3 = (self.x_conv2
                    .sg_conv(dim=256)
                    .sg_conv()
                    .sg_conv()
                    .sg_conv()
                    .sg_pool()) # (1, 28, 28, 256)
            self.x_conv4 = (self.x_conv3
                    .sg_conv(dim=512)
                    .sg_conv()
                    .sg_conv()
                    .sg_conv()
                    .sg_pool()) # (1, 14, 14, 512)
#                     .sg_conv(dim=512)
#                     .sg_conv()
#                     .sg_conv()
#                     .sg_conv()
#                     .sg_pool())

        self.y_conv1 = self.x_conv1.sg_reuse(input=self.y)
        self.y_conv2 = self.x_conv2.sg_reuse(input=self.y)
        self.y_conv3 = self.x_conv3.sg_reuse(input=self.y)
        self.y_conv4 = self.x_conv4.sg_reuse(input=self.y)
#  
        def get_gram_mat(tensor):
            '''
            Arg:
              tensor: 4-D tensor. The first  dimension must be 1.
            
            Returns:
              gram matrix. Read `https://en.wikipedia.org/wiki/Gramian_matrix` for details.
              512 by 512.
            '''
            assert tensor.get_shape().ndims == 4, "The tensor must be 4 dimensions."
            
            dim0, dim1, dim2, dim3 = tensor.get_shape().as_list()
            tensor = tensor.sg_reshape(shape=[dim0*dim1*dim2, dim3]) #(1*7*7, 512)
            
            # normalization: Why? Because the original value of gram mat. would be too huge.
            mean, variance = tf.nn.moments(tensor, [0, 1])
            tensor = (tensor - mean) / tf.sqrt(variance + tf.sg_eps)
            
            tensor_t = tensor.sg_transpose(perm=[1, 0]) #(512, 1*7*7)
            gram_mat = tf.matmul(tensor_t, tensor) # (512, 512)
               
            return gram_mat
        
        # Loss: Add the loss of each layer
        self.mse = tf.squared_difference(get_gram_mat(self.x_conv1), get_gram_mat(self.y_conv1)).sg_mean() +\
                   tf.squared_difference(get_gram_mat(self.x_conv2), get_gram_mat(self.y_conv2)).sg_mean() +\
                   tf.squared_difference(get_gram_mat(self.x_conv3), get_gram_mat(self.y_conv3)).sg_mean() +\
                   tf.squared_difference(get_gram_mat(self.x_conv4), get_gram_mat(self.y_conv4)).sg_mean()
                   
        self.train_gen = tf.sg_optim(self.mse, lr=0.0001, category='generator')  # Note that we train only variable x.
        
def generate(sample_image): 
    start_time = time.time() 
 
    g = ModelGraph()
         
    with tf.Session() as sess:
        # We need to initialize variables in this case because the Variable `generator/x` will not restored.
        tf.sg_init(sess)
         
        vars = [v for v in tf.global_variables() if "generator" not in v.name]
        saver = tf.train.Saver(vars)
        saver.restore(sess, tf.train.latest_checkpoint('asset/train/ckpt'))
          
        i = 0
        while True:
            mse, _ = sess.run([g.mse, g.train_gen], {g.y: transform_image(sample_image)}) # (16, 28)
               
            if time.time() - start_time > 60: # Save every 60 seconds
                gen_image = sess.run(g.x)
                gen_image = np.squeeze(gen_image)
                misc.imsave('gen_images/%s/gen_%.2f.jpg' % (label, mse), gen_image)
                   
                start_time = time.time()
                i += 1
                if i == 60: break # Finish after 1 hour

if __name__ == '__main__':
    label = sample_image.split("/")[-1].split("-")[0]
    if not os.path.exists("gen_images/" + label): os.makedirs("gen_images/" + label)  
                  
    # Save cropped image of the target image.
    misc.imsave("gen_images/{}/{}.jpg".format(label, label), np.squeeze(transform_image(sample_image)))  
          
    generate(sample_image)
    print "Done"


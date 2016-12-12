# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Training.
'''
from prepro import Data
import sugartensor as tf
import numpy as np
import random

class ModelGraph():
    '''Builds a model graph'''
    def __init__(self):
        '''
        Args:
          is_train: Boolean. If True, backprop is executed.
        '''
        train_data = Data() # (16, 224, 224, 1), (16,)
        
        self.x = train_data.x
        self.y = train_data.y
        self.idx2label = train_data.idx2label
        self.label2idx = train_data.label2idx
        
        self.conv5 = self.x.sg_vgg_19(conv_only=True) # (batch_size, 7, 7, 512)
        
        self.logits = (self.conv5.sg_conv(size=1, stride=1, dim=28, act="linear", bn=False)
                  .sg_mean(dims=[1, 2], keep_dims=False) )# (16, 28)
        
        self.ce = self.logits.sg_ce(target=self.y, mask=False) # (16,)
        
        # training accuracy
        self.acc = (self.logits.sg_softmax()
                               .sg_accuracy(target=self.y, name='training_acc'))
            
            
def train():
    g = ModelGraph(); print "Graph loaded!"
    
    tf.sg_train(lr=0.00001, lr_reset=True, log_interval=10, loss=g.ce, eval_metric=[g.acc], max_ep=5, 
                    save_dir='asset/train', early_stop=False, max_keep=5)

if __name__ == '__main__':
    train(); print "Done"

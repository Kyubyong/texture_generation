# -*- coding: utf-8 -*-

import sugartensor as tf
import numpy as np
from train import ModelGraph
import codecs

def main():  
    g = ModelGraph()
        
    with tf.Session() as sess:
        tf.sg_init(sess)

        # restore parameters
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('asset/train/ckpt'))
        
        hits = 0
        num_imgs = 0
        
        with tf.sg_queue_context(sess):
            # loop end-of-queue
            while True:
                try:
                    logits, y = sess.run([g.logits, g.y]) # (16, 28) 
                    preds = np.squeeze(np.argmax(logits, -1)) # (16,)
                     
                    hits += np.equal(preds, y).astype(np.int32).sum()
                    num_imgs += len(y)
                    print "%d/%d = %.02f" % (hits, num_imgs, float(hits) / num_imgs)
                except:
                    break
                
        print "\nFinal result is\n%d/%d = %.02f" % (hits, num_imgs, float(hits) / num_imgs)
                    
                    
                             
#                     fout.write(u"▌file_name: {}\n".format(f))
#                     fout.write(u"▌Expected: {}\n".format(label2cls[]))
#                     fout.write(u"▌file_name: {}\n".format(f))
#                     fout.write(u"▌Got: " + predicted + "\n\n")
                                        
if __name__ == '__main__':
    main()
    print "Done"


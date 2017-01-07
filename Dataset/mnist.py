"""MNIST Dataset

This comes from David Ha. https://github.com/hardmaru/supercell/blob/master/assets/MNIST_Static_HyperNetwork_Example.ipynb
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from IPython import embed
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class Dataset(object):
    """A Class used to store mnist data"""

    def __init__(self, images, labels, augment=False, seed=2014):
        images = images.astype(np.float32)
        self.image_size = 28
        self._num_examples = len(images)
        images = np.reshape(images, (self._num_examples, self.image_size, self.image_size, 1)) # (N, 28, 28 , 1)
        
        perm = np.arange(self._num_examples)
        np.random.seed(seed)
        np.random.shuffle(perm)
        self._images = images[perm]
        self._labels = labels[perm]
        self._augment = augment
        self.pointer = 0
        self.upsize = 1 if self._augment else 0
        self.min_upsize = 2
        self.max_upsize = 2
        self.random_perm_mode = False
        self.num_classes = 10 # mnist contains 10 classes
    
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def num_examples(self):
        return self._num_examples
    
    def next_batch(self, batch_size=100, with_label=True, one_hot = False):
        if self.pointer >= self.num_examples-2*batch_size: # reset pointer to 0 if reach data end
            self.pointer = 0
        else:
            self.pointer += batch_size
        result = []
        
        # always upsize 2 
        upsize_amount = np.random.randint(self.upsize*self.min_upsize, self.upsize*self.max_upsize+1) 
        
        def upsize_row_once(img):
            """random add one row in image and fill it using the average of neighbors"""
            old_size = img.shape[0]
            new_size = old_size + 1
            new_img = np.zeros((new_size, img.shape[1], 1))
            rand_row = np.random.randint(1, old_size-1)
            new_img[0:rand_row,:] = img[0:rand_row,:]
            new_img[rand_row+1:,:] = img[rand_row:,:]
            new_img[rand_row,:] = 0.5 * (new_img[rand_row-1,:] + new_img[rand_row+1,:])
            return new_img
        def upsize_col_once(img):
            """random add one col in image and fill it using the average of neighbors"""
            old_size = img.shape[1]
            new_size = old_size +1
            new_img = np.zeros((img.shape[0], new_size, 1))
            rand_col = np.random.randint(1,old_size-1)
            new_img[:,0:rand_col] = img[:,0:rand_col]
            new_img[:,rand_col+1:] = img[:,rand_col:]
            new_img[:,rand_col] = 0.5 * (new_img[:,rand_col-1] + new_img[:,rand_col+1])
            return new_img
        def upsize_me(img,n=self.max_upsize):
            new_img = img
            for i in range(n):
                new_img = upsize_row_once(new_img)
                new_img = upsize_col_once(new_img)
            return new_img

        for data in self._images[self.pointer:self.pointer+batch_size]:
            result.append(self.crop_image(upsize_me(data, upsize_amount),upsize_amount))

        if len(result) != batch_size:
            raise ValueError("at pointer {}, number of upsized images must be the same as raw images. ".format(self.pointer))
        
        result_labels = self.labels[self.pointer:self.pointer+batch_size]
        
        #convert label to one-hot
        if one_hot:
            result_labels = np.eye(self.num_classes)[result_labels]
        if with_label:
            return np.array(result, dtype=np.float32), result_labels
        return np.array(result, dtype=np.float32)
        
    def crop_image(self, img, upsize_amount):
        """random crop the image to size = <size-upsize_amount, size-upsize_amount>
        
        This is used to random crop the upsized images(30,30) back to original image size(28,28).
        """
        row_distort = np.random.randint(0,upsize_amount+1)
        col_distort = np.random.randint(0,upsize_amount+1)
        cropped_img = np.zeros((self.image_size, self.image_size,1),dtype=np.float32)
        cropped_img[:,:,:] = img[row_distort:row_distort+self.image_size,col_distort:col_distort+self.image_size,:]
        return cropped_img


def read_data_sets(mnist_data=None):
    if mnist_data is None:
        mnist_data = input_data.read_data_sets('/tmp/data/',one_hot=False)
    class DataSets(object):
        pass
    data_sets = DataSets()
    data_sets.train = Dataset(mnist_data.train.images, mnist_data.train.labels, augment=True)
    data_sets.valid = Dataset(mnist_data.validation.images, mnist_data.validation.labels, augment=False)
    data_sets.test = Dataset(mnist_data.test.images, mnist_data.test.labels, augment=False)
    return data_sets
        

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    batch_size = 100
    mnist_data = input_data.read_data_sets('/tmp/data/',one_hot=False)
    data = read_data_sets(mnist_data)
    batch = data.train.next_batch(batch_size=batch_size,one_hot=True)
    plt.imshow(batch[0][np.random.randint(0,batch_size+1),:,:,0])
    plt.gray()
    plt.show()
    embed()

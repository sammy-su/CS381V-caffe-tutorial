
# coding: utf-8

# # Set up network

# First, import required modules, set plotting parameters, and run `./scripts/download_model_binary.py models/bvlc_reference_caffenet` to get the pretrained CaffeNet model if it hasn't already been fetched.

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# Make sure that caffe is on the python path:
caffe_root = '/Users/dineshj/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os
if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")
    get_ipython().system(u'/Users/dineshj/caffe/scripts/download_model_binary.py /Users/dineshj/caffe/models/bvlc_reference_caffenet')


# Set Caffe to CPU mode, load the net in the test phase for inference.

# In[2]:

caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)


# # Access weights

# Just like net.blobs for features, net.params stores the weights. 

# In[3]:

[(k, v[0].data.shape) for k, v in net.params.items()]


# In[4]:

# the parameters are a list of [weights, biases]
np.shape(net.params['fc8'][0].data)


# # Visualize convolutional filters

# Helper functions for visualization

# In[5]:

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)


# In[6]:

filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))


# The second layer filters, `conv2`
# 
# There are 256 filters, each of which has dimension 5 x 5 x 48. We show only the first 48 filters, with each channel shown separately, so that each filter is a row.

# In[7]:

filters = net.params['conv2'][0].data
vis_square(filters[:48].reshape(48**2, 5, 5))


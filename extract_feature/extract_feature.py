#!/usr/bin/env python
# encoding: utf-8

import argparse
import numpy as np
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["GLOG_minloglevel"] = "3"
caffe_root = '/work/01932/dineshj/CS381V/caffe_install_scripts/caffe'
pycaffe_dir = os.path.join(caffe_root, 'python')
sys.path.append(pycaffe_dir)

import caffe

class Extractor(caffe.Net):
    def __init__(self, model_file, pretrained_file, mean_file):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        mean = np.load(mean_file)
        if mean.shape[1:] != (1, 1):
            mean = mean.mean(1).mean(1)

        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer({in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2,0,1))
        self.transformer.set_mean(in_, mean)
        self.transformer.set_channel_swap(in_, (2,1,0))
        self.features = ['fc7']

    def set_features(self, feature_layer):
        self.features = feature_layer

    def extract_features(self, images):
        in_ = self.inputs[0]
        caffe_in = np.zeros((len(images),)+self.blobs[in_].data.shape[1:])
        for ix, image in enumerate(images):
            caffe_in[ix] = self.transformer.preprocess(in_, caffe.io.load_image(image))
        features = self.forward_all(**{in_: caffe_in, 'blobs': self.features})
        return features

    def extract_feature(self, image):
        in_ = self.inputs[0]
        self.blobs[in_].data[...] = self.transformer.preprocess(in_, caffe.io.load_image(image[0]))
        feature = self.forward(**{'blobs': self.features})
        feature = {blob: vals[0] for blob, vals in feature.iteritems()}
        return feature

def extractor_factory():
    model_def = os.path.join(caffe_root, "models/bvlc_reference_caffenet/deploy.prototxt")
    pretrained_model = os.path.join(caffe_root, "models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel")
    mean_file = os.path.join(pycaffe_dir, 'caffe/imagenet/ilsvrc_2012_mean.npy')

    extractor = Extractor(model_def, pretrained_model, mean_file)
    return extractor

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
    plt.savefig("visualize_conv.jpg")
    print("conv1 filter visualizations saved to visualize_conv.jpg")
    plt.close()

def vis_fc(feature):
    plt.plot(feature)
    plt.xlim(xmax=feature.shape[0])
    plt.savefig("visualize_fc.jpg")
    print("fc7 feature plot saved to visualize_fc.jpg")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_def",
        default=os.path.join(caffe_root,
            "models/bvlc_reference_caffenet/deploy.prototxt"),
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(caffe_root,
            "models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
            'caffe/imagenet/ilsvrc_2012_mean.npy'),
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "image",
        nargs=1,
        help="Image file to be processed."
    )

    args = parser.parse_args()

    if args.gpu:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    extractor = Extractor(args.model_def, args.pretrained_model, args.mean_file)

    feature = extractor.extract_feature(args.image)
    vis_fc(feature['fc7'])
    filters = extractor.params['conv1'][0].data
    vis_square(filters.transpose(0,2,3,1))
    np.save('feature.npy', feature);
    print("All features saved to feature.npy");

if __name__ == "__main__":
    main()

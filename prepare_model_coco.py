import os
import pdb
from sys import stdout
import scipy
import  cPickle as pickle
import numpy as np
# import matplotlib.pyplot as plt
#matplotlib inline
import sys
#sys.path.insert(0, caffe_root + 'python')
import caffe
# plt.rcParams['figure.figsize'] = (10, 10)
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
import os
import pandas as pd
import nltk
import pdb

coco_dir = "/home/gy46/arctic-captions/data/coco/"
caffe_dir = '/home/gy46/caffe/'
vgg_dir = os.path.join(caffe_dir, 'models', 'vgg_ilsvrc_19')

#Setup
print 'Start Setup'
trainImagesPath = os.path.join(coco_dir, 'train2014')
valImagesPath = os.path.join(coco_dir, 'val2014')
testImagesPath = os.path.join(coco_dir, 'test2014')

imagesPaths = {
    'train' : trainImagesPath,
    'val'   : valImagesPath,
    'test'  : testImagesPath
}

vgg_ilsvrc_19_layoutFileName = os.path.join(vgg_dir, 'VGG_ILSVRC_19_layers_deploy.prototxt')
vgg_ilsvrc_19_modelFileName = os.path.join(vgg_dir, 'VGG_ILSVRC_19_layers.caffemodel')
dataPath = coco_dir
annotation_path = os.path.join(coco_dir, 'annotations', 'captions_train2014.json')
experimentPrefix = '.exp1'
print 'End Setup'

#Setup caffe
print 'Start Setup caffe'
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(vgg_ilsvrc_19_layoutFileName,vgg_ilsvrc_19_modelFileName,caffe.TEST)
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
mean_image = os.path.join(caffe_dir, 'python', 'caffe', 'imagenet', 'ilsvrc_2012_mean.npy')
# transformer.set_mean('data', np.load('/tmp3/alvin/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_mean('data', np.load(mean_image).mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
print 'End Setup caffe'

#filelist:
#./splits/coco_val.txt
#./splits/coco_test.txt
#./splits/coco_train.txt

# set net to batch size of 50
# net.blobs['data'].reshape(10,3,224,224)
print 'Start middle'
files = [ 'val','test']
# files = [ 'val' ]
# files = [ 'test' ]

# for fname in files:
# for fname in ['test']:
for fname in []:
    print fname
    f = open('./splits/coco_'+fname+'.txt')
    counter = 0
    imageList = [i for i in f]
    numImage = len(imageList)
    result = np.memmap('tmp.np.array', dtype=np.float64, mode='w+', shape=(numImage, 100352))

    for i in range(numImage):
        fn = imageList[i].rstrip()
        img_p = os.path.join(imagesPaths[fname], fn)
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img_p))
        out = net.forward()
        feat = net.blobs['conv5_4'].data[0]
        reshapeFeat = np.swapaxes(feat,0,2)
        reshapeFeat2 = np.reshape(reshapeFeat,(1,-1))
        counter += 1
        stdout.write("\r%d/%d" % (counter, numImage))
        stdout.flush()
        result[i,:] = reshapeFeat2
    print result.shape

    resultSave = scipy.sparse.csr_matrix(result)
    resultSave32 = resultSave.astype('float32')
    np.savez(dataPath + 'coco_feature.' + fname + experimentPrefix, data=resultSave32.data, indices=resultSave32.indices, indptr=resultSave32.indptr, shape=resultSave.shape)

print 'End middle'

print 'Start end'
#np.savez(dataPath + 'coco_feature.' + fname + experimentPrefix, data=resultSave32.data, indices=resultSave32.indices, indptr=resultSave32.indptr, shape=resultSave.shape)

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices, indptr =array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

capDict = pickle.load(open('capdict.pkl','rb'))

for name in files:
    counter = 0
    filenames = open('./splits/coco_'+name+'.txt')
    cap = []
    for imageFile in filenames:
        imageFile = imageFile.rstrip()
        if name != 'test':
            for sen in capDict[imageFile]:
                cap.append([sen.rstrip(), counter])
        else:
            for _ in range(5):
                cap.append(["", counter])
        counter += 1
    saveFile = open(dataPath + 'coco_align.' + name + experimentPrefix + '.pkl', 'wb')
    pickle.dump(cap, saveFile, protocol=pickle.HIGHEST_PROTOCOL)
    saveFile.close()
    filenames.close()

#print wordsDict['Two']
#print resultSave32

print 'Start end'









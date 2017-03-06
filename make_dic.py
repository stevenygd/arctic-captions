import  cPickle as pickle
import numpy as np
import pandas as pd
import nltk
import os

coco_dir = "/home/gy46/arctic-captions/data/coco/"
annotation_path= os.path.join(coco_dir, 'captions.token')
# annotation_path = "/tmp3/alvin/arctic_data/coco/captions.token"
annotations = pd.read_table(annotation_path, sep='\t', header=None,names=['image', 'caption'])
captions = annotations['caption'].values
words = nltk.FreqDist(' '.join(captions).split()).most_common()
wordsDict = {words[i][0]:i+2 for i in range(len(words))}
# with open('/tmp3/alvin/arctic_data/coco/dictionary.pkl', 'wb') as f:
with open(os.path.join(coco_dir, 'dictionary.pkl'), 'wb') as f:
    pickle.dump(wordsDict, f)

print words[:10]

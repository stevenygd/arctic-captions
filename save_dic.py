from os import listdir
import os
import json
import cPickle


coco_dir = "/home/ec2-user/captions/l-arctic-captions/data/coco/"
coco_annotations = os.path.join(coco_dir, 'annotations')

def rename_train(s):
    return 'COCO_train2014_'+'0'*(12-len(s))+s+'.jpg'

def rename_val(s):
    return 'COCO_val2014_'+'0'*(12-len(s))+s+'.jpg'

dic={}

with open(os.path.join(coco_annotations, 'captions_train2014.json')) as fp:
# with open('/tmp3/alvin/dataset/MSCOCO2014/captions_train2014.json') as fp:
    data = json.load(fp)

cnt=0
for i in range(len(data['annotations'])):
    s = rename_train(str(data['annotations'][i]['image_id']))
    if s not in dic:
        dic[s]=[]
        dic[s].append(data['annotations'][i]['caption'].replace('\n','').replace('"',''))
        cnt+=1
    else:
        #if len(dic[s])==5:
        #    continue
        dic[s].append(data['annotations'][i]['caption'].replace('\n','').replace('"',''))
        cnt+=1
print cnt


with open(os.path.join(coco_annotations, 'captions_val2014.json')) as fp:
# with open('/tmp3/alvin/dataset/MSCOCO2014/captions_val2014.json') as fp:
    data = json.load(fp)

for i in range(len(data['annotations'])):
    s = rename_val(str(data['annotations'][i]['image_id']))
    if s not in dic:
        dic[s]=[]
        dic[s].append(data['annotations'][i]['caption'].replace('\n','').replace('"',''))
        cnt+=1
    else:
        #if len(dic[s])==5:
        #    continue
        dic[s].append(data['annotations'][i]['caption'].replace('\n','').replace('"',''))
        cnt+=1
print cnt

with open('capdict.pkl','wb') as fp:
    cPickle.dump(dic,fp,protocol=cPickle.HIGHEST_PROTOCOL)



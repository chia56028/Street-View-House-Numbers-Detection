import os
import cv2
import json

filepath = 'runs/detect/exp16/labels/'

data = []
for i in range(1, 13069):
    if not os.path.isfile(filepath+str(i)+'.txt'):
        a = {"bbox": [(1,1,1,1)], "score": [0.5], "label": [0]}
    else:
        f = open(filepath+str(i)+'.txt','r')
        contents = f.readlines()

        img_name = str(i)+'.png'
        im = cv2.imread('data/svhn/test/'+img_name)
        h, w, c = im.shape
        # print(h,w)
        
        a = {"bbox": [], "score": [], "label": []}
        for content in contents:
            content = content.replace('\n','')
            c = content.split(' ')
            # print(c)
            a['label'].append(int(c[0]))
            w_center = w*float(c[1])
            h_center = h*float(c[2])
            width = w*float(c[3])
            height = h*float(c[4])
            left = int(w_center - width/2)
            right = int(w_center + width/2)
            top = int(h_center - height/2)
            bottom = int(h_center + height/2)
            a['bbox'].append(tuple((top, left, bottom, right)))
            a['score'].append(float(c[5]))

    # print(a)
    data.append(a)
    f.close()

ret = json.dumps(data)

print(len(data))
with open('0856172_2.json', 'w') as fp:
    fp.write(ret)
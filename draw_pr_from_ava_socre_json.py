import json
CLASSES = ['__background__',  # always index 0
           'tibetan flag', 'guns','knives','not terror',
'islamic flag','isis flag']


score_file = '111OHEM_vali_score.txt'

thresh_old = 0.9

fout = open('res.txt','w')

with open(score_file, 'rb') as fid:
    all_boxes  = fid.readlines()
print(len(all_boxes))

y_true =[]
y_scores = []
for i in range(0,len(all_boxes)):
    res_dict = json.loads(all_boxes[i])
    filename = res_dict['url']
#    for h in xrange
    if filename.strip().split("/")[-1][0] == '1':
        y_true.append(1)
    else:
        y_true.append(0)
    max_score =0
    bboxes = res_dict['label']['detect']['general_d']['bbox']
    for box in bboxes:
        if box['class'] == 'not terror':
            continue
        det_score = box['score']
        box_pt = box['pts']
        if det_score > max_score:
            max_score = det_score
        if det_score > thresh_old:
            line = filename.strip() + ' ' + box['class'] + ' ' + str(det_score) + ' ('+str(box_pt[0]) + ',' + str(box_pt[1]) + ','+str(box_pt[2])+','+ str(box_pt[3])+')\n'
            fout.write(line)
#    print(boxes_this_image)
    y_scores.append(max_score)
fout.close()

import numpy as np
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

precision, recall, _ = precision_recall_curve(y_true, y_scores)
max_f1 =0
for i in range(0,len(precision)):
    F1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
    if F1 >max_f1:
        max_f1 = F1

print("max_f1 = " + str(max_f1))
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_true, y_scores)
print("ap = " + str(average_precision))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(average_precision))
plt.savefig('1.jpg')

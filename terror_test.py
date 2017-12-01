import numpy as np
import sys,os
import cv2
caffe_root = '/workspace/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import random

#net_file= 'MobileNetSSD_deploy.prototxt'
#caffe_model='MobileNetSSD_deploy.caffemodel'
#test_dir = "images"
def net_init(args):
    if not os.path.exists(args.caffe_model):
        print("MobileNetSSD_deploy.affemodel does not exist,")
        print("use merge_bn.py to generate it.")
        exit()
    net = caffe.Net(args.net_file,args.caffe_model,caffe.TEST)
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
    return net

CLASSES = ('background',
           'tibetan flag', 'guns', 'knives', 'not terror',
           'islamic flag', 'isis flag')


def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
#    print(out['detection_out'].shape)
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(net,image_file,args):
#    origimg = cv2.imread(imgfile)
    import urllib
    proxies = {'http' : 'http://xsio.qiniu.io'}
    data = urllib.urlopen(image_file.strip(),proxies=proxies).read()
    nparr = np.fromstring(data, np.uint8)
    origimg = cv2.imdecode(nparr,1)
    img = preprocess(origimg)
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()
    box, conf, cls = postprocess(origimg, out)

    if args.vis:
        for i in range(len(box)):
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
            p1 = (box[i][0], box[i][1])
            p2 = (box[i][2], box[i][3])
            cv2.rectangle(origimg, p1, p2, color=color,thickness=2)
            p3 = (p1[0], p1[1]+15)
            title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
            cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (255, 255, 255), 1)
        vis_image_dir = 'vis'
        result_file = os.path.join(vis_image_dir, image_file.strip().split('/')[-1] + '_result' + '.JPEG')
        cv2.imwrite(result_file, origimg)
    return box,conf,cls

def generate_ava_json(image_file,box,conf,cls,thresh):
    dets, dets_s = [], []
    for index,box_now in enumerate(box):
        if CLASSES[int(cls[index])] == 'not terror':
            continue
        if conf[index] > thresh:
            det, det_s = dict(), dict()
            xmin = float(box[index][0])
            ymin = float(box[index][1])
            xmax = float(box[index][2])
            ymax = float(box[index][3])
            det['pts'] = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            det['class'] = CLASSES[int(cls[index])]
            det_s['pts'] = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            det_s['class'] = CLASSES[int(cls[index])]
            det_s['score'] = float(conf[index])

            dets.append(det)
            dets_s.append(det_s)
        #line = {}
        #line['detections'] = dets
        #line['img'] = image_file

    ress = {
        "url": image_file,
        "label": {"detect": {"general_d": {"bbox": dets}}},
        "type": "image",
        "source_url": "",
        "ops": "download()"
    }
    ress_s = {
        "url": image_file,
        "label": {"detect": {"general_d": {"bbox": dets_s}}},
        "type": "image",
        "source_url": "",
        "ops": "download()"
    }

    return ress , ress_s

def test_file_list(net,args):
    fout = open(args.out_prefix + '_vali.txt','w')
    fout_score = open(args.out_prefix + '_vali_score.txt', 'w')
    fout_miss = open(args.out_prefix + '_vali_miss.txt', 'w')
    assert os.path.exists(args.test_file), args.test_file + ' not found'
    with open(args.test_file) as f:
        image_set_index = [x.strip().split(' ')[0] for x in f.readlines()]
    num_images = len(image_set_index)
    for index , image_file in enumerate(image_set_index):
        print("processing {}/{} image:{}".format(index, num_images, image_file))
        box,conf,cls = detect(net,image_file,args)
        ress,ress_s = generate_ava_json(image_file,box,conf,cls,args.thresh)
        import json 
        if len(ress['label']['detect']['general_d']['bbox'])>=0:
            fout.write(json.dumps(ress)+'\n')
            fout.flush()
            fout_score.write(json.dumps(ress_s)+'\n')
            fout_score.flush()
            #print(json.dumps(ress))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Test a SSD network')
    # general
    parser.add_argument('--gpu', help='gpu id', required=True, type=int)
    parser.add_argument('--net_file',help='caffe prototxtfile',
                        default='MobileNetSSD_deploy.prototxt',type=str)
    parser.add_argument('--caffe_model',help='caffe modelfile',
                        default='MobileNetSSD_deploy.caffemodel',type=str)
    parser.add_argument('--test_file', help='test file list', required=True, type=str)
    parser.add_argument('--out_prefix', help='output prefix', required=True, type=str)

    # rcnn
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    # parser.add_argument('--use_box_voting', help='use box voting in test', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    net = net_init(args)
    if net is None :
        print("net inti error")
        exit()
    test_file_list(net,args)


from __future__ import division, print_function, absolute_import

import cv2
from base_camera import BaseCamera

import warnings
import numpy as np
from PIL import Image
from importlib import import_module
from collections import Counter
import datetime

import argparse
import time, datetime
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *

from reid.data import make_data_loader
from reid.data.transforms import build_transforms
from reid.modeling import build_model
from reid.config import cfg as reidCfg
import map
warnings.filterwarnings('ignore')

class Camera(BaseCamera):
    def __init__(self, feed_type, device, port_list):
        super(Camera, self).__init__(feed_type, device, port_list)

    @staticmethod
    def yolo_frames(unique_name):
        cam_id = unique_name[1]
        device = torch_utils.select_device(force_cpu=False)

        torch.backends.cudnn.benchmark = False  # set False for reproducible results
        
        cfg='cfg/yolov3.cfg'
        data='data/coco.data'
        weights='weights/yolov3.weights'
        half=False
        img_size=416
        conf_thres=0.25
        nms_thres=0.4
        dist_thres=1.3

        #reid 모델 생성
        query_loader, num_query = make_data_loader(reidCfg)
        reidModel = build_model(reidCfg, num_classes=10126)
        reidModel.load_param(reidCfg.TEST.WEIGHT)
        reidModel.to(device).eval()

        #reid하기 위한 query 정보
        query_feats = defaultdict(list)
        query_pids = []

        #query 정보 가져오기
        for i, batch in enumerate(query_loader):
            with torch.no_grad():
                img, pid, camid = batch
                img = img.to(device)
                feat = reidModel(img)         
                for j,f in enumerate(feat):
                    if(not pid[j] in query_pids):
                        query_pids.append(pid[j])
                    print(f.cpu().numpy())
                    query_feats[pid[j]].append(f.cpu().numpy())

        #query 정보 torch형식으로 변경
        for pid in query_pids:
            temp = np.array(query_feats[pid])
            print(temp)
            query_feats[pid] = torch.from_numpy(temp).float().to(device) 
            print(query_feats[pid])
            query_feats[pid] = torch.nn.functional.normalize(query_feats[pid], dim=1, p=2) 
            print(query_feats[pid])
        print("The query feature is normalized") 

        
        model = Darknet(cfg, img_size) #config로 디텍션 모델 생성

        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        # Eval mode
        model.to(device).eval()
        # Half precision
        half = half and device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            model.half()

        # Set Dataloader
        print("cam id is "+cam_id)
        dataloader = LoadWebcam(cam_id, img_size=img_size, half=half)

        # Get classes and colors
        # parse_data_cfg(data)['names'] names=data/coco.names
        classes = load_classes(parse_data_cfg(data)['names']) # 코코 네임 파일에서 클래스 전부 가져옴 ['person', 'bicycle'...]
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))] # 이 클래스 수대로 박스의 색을 random으로 생성
    
        #count = 0 #이미지 자르기 위한 카운트
        # Run inference
        t0 = time.time()
        for i, (path, img, im0, vid_cap) in enumerate(dataloader):
            patient_map = map.Map
            now = time.localtime()

            # If the saved minute is not same with the current minute, it means this camera is the first one that access to the map information in this minute.
            # Since we have to clear the second array every minute, change the minute and clear the second array.
            if(now.tm_min != patient_map.minute):
                patient_map.minute = now.tm_min
                patient_map.sec_array.clear()

            # If there is no information about current second, it means this is the first access to the map in this second.
            # We should init the map information each second.    
            if(now.tm_sec not in patient_map.sec_array):
                patient_map.sec_array.append(now.tm_sec)
                patient_map.camera_map = {0:False, 1:False, 2:False}

            if i % 3 != 0: #이미지 처리 부하 줄이기
                continue

            # Get detections shape: (3, 416, 320)
            img = torch.from_numpy(img).unsqueeze(0).to(device) # torch.Size([1, 3, 416, 320]) #이미지 torch 형식으로 바꾸기
            pred, _ = model(img) #이미지 디텍션
            det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0] # torch.Size([5, 7]) #threshold로 이미지 거른 후 det변수에 초기화

            if det is not None and len(det) > 0:
                # Rescale boxes from 416 to true image size 
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results to screen image 1/3 data\samples\000493.jpg: 288x416 5 persons, Done. (0.869s)
                #print('%gx%g ' % img.shape[2:], end='')  # print image size '288x416'
                #for c in det[:, -1].unique():   # 对图片的所有类进行遍历循环
                #   n = (det[:, -1] == c).sum() # 得到了当前类别的个数，也可以用来统计数目
                    #if classes[int(c)] == 'person':
                    #    print('%g %ss' % (n, classes[int(c)]), end=', ') # 打印个数和类别'5 persons'
                    #    print(" ")

                # Draw bounding boxes and labels of detections
                # (x1y1x2y2, obj_conf, class_conf, class_pred)
                
                gallery_img = [] #사람의 이미지만 따로 저장하는 Array
                gallery_loc = [] #사진의 좌표를 저장하는 Array
                for *xyxy, conf, cls_conf, cls in det: # det의 정보엔 박스의 좌표, 정확도, 클래스인지 확인하는 정확도가 있음
                    # *xyxy: [tensor(349.), tensor(26.), tensor(468.), tensor(341.)] tensor로 좌표가 있음
                    

                    # Add bbox to the image
                    # label = '%s %.2f' % (classes[int(cls)], conf) # 'person 1.00'
                    if classes[int(cls)] == 'person': # detect한 클래스가 person class라면
                        xmin = int(xyxy[0])
                        ymin = int(xyxy[1])
                        xmax = int(xyxy[2])
                        ymax = int(xyxy[3])
                        w = xmax - xmin # 233
                        h = ymax - ymin # 602
                        if w*h > 500:
                            gallery_loc.append((xmin, ymin, xmax, ymax))
                            crop_img = im0[ymin:ymax, xmin:xmax] # HWC (602, 233, 3)
                            #cv2.imwrite('./temp/'+str(count)+'.jpg',crop_img) #query 이미지로 쓰기위해 temp 폴더에 내 이미지 저장
                            #count=count+1
                            crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))  # PIL: (233, 602)
                            crop_img = build_transforms(reidCfg)(crop_img).unsqueeze(0)  # torch.Size([1, 3, 256, 128])
                            gallery_img.append(crop_img)
                            plot_one_box(xyxy, im0, color=[128,128,128]) #사람을 흰색으로 박스 치기

                if gallery_img: #사람의 이미지만 자른 Array의 데이터가 존재하면
                    gallery_img = torch.cat(gallery_img, dim=0)  # torch.Size([7, 3, 256, 128])
                    gallery_img = gallery_img.to(device)
                    gallery_feats = reidModel(gallery_img) # torch.Size([7, 2048])
                    #print("The gallery feature is normalized")
                    gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)  # 计算出查询图片的特征向量
                    #그 이미지의 특징을 뽑아옴

                    # m: 2
                    # n: 7
                    for pid in query_pids:
                        m, n = query_feats[pid].shape[0], gallery_feats.shape[0]
                        distmat = torch.pow(query_feats[pid], 2).sum(dim=1, keepdim=True).expand(m, n) + \
                                  torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                        # out=(beta∗M)+(alpha∗mat1@mat2)
                        # qf^2 + gf^2 - 2 * qf@gf.t()
                        # distmat - 2 * qf@gf.t()
                        # distmat: qf^2 + gf^2
                        # qf: torch.Size([2, 2048])
                        # gf: torch.Size([7, 2048])
                        distmat.addmm_(1, -2, query_feats[pid], gallery_feats.t())
                        # distmat = (qf - gf)^2
                        # distmat = np.array([[1.79536, 2.00926, 0.52790, 1.98851, 2.15138, 1.75929, 1.99410],
                        #                     [1.78843, 1.96036, 0.53674, 1.98929, 1.99490, 1.84878, 1.98575]])
                        distmat = distmat.cpu().detach().numpy()  # <class 'tuple'>: (3, 12)
                        distmat = distmat.sum(axis=0) / len(query_feats[pid]) # 쿼리의 특징과 현재 이미지의 특징의 차이를 계산
                        index = distmat.argmin()
                        if distmat[index] < dist_thres: #그 차이가 위에서 지정한 treshold보다 작으면 일치하다
                            print('목표 찾음 %s번 카메라：%s'%(cam_id,distmat[index]))                                    
                            plot_one_box(gallery_loc[index], im0, label='find!', color=[0,0,255])

                            #If the map of this camera ID is still false, it means there was no identified query in this second.
                            if(patient_map.camera_map[int(cam_id)] == False):
                                patient_map.camera_map[int(cam_id)] = True
                                filename = time.strftime("%Y%m%d", time.localtime(time.time())) + '_c'+cam_id+'.txt'
                                f = open(filename, 'a')
                                f.write('\n'+time.strftime('%H : %M : %S'))
                                f.close
                
                yield cam_id, im0

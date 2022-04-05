# import argparse
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from gaze_utils import select_device, draw_gaze
from PIL import Image, ImageOps

from face_detection import RetinaFace
from model import L2CS

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
face = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.2)
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils


def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model

class Gaze:
    def __init__(self):
        cudnn.enabled = True
        arch = 'ResNet50'
        batch_size = 1
        self.gpu = select_device("0", batch_size=batch_size)
        snapshot_path = './gaze_direction/models/L2CSNet_gaze360.pkl'

        self.transformations = transforms.Compose([
                transforms.Resize(448),
                transforms.ToTensor(),
                transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
    
        self.model = getArch(arch, 90)
        print('Loading gaze direction snapshot.')
        saved_state_dict = torch.load(snapshot_path)
        self.model.load_state_dict(saved_state_dict)
        self.model.cuda(self.gpu)
        self.model.eval()

        self.softmax = nn.Softmax(dim=1)
        self.detector = RetinaFace(gpu_id=0)
        idx_tensor = [idx for idx in range(90)]
        self.idx_tensor = torch.FloatTensor(idx_tensor).cuda(self.gpu)
        x=0
        
    def find_gaze(self, frame, imgRGB, adult_loc, child_loc):
        faces = self.detector(frame)
        if faces is not None:
            child_feats = {'pitch': [], 'yaw': []}
            adult_feats = {'pitch': [], 'yaw': []}
            for box, landmarks, score in faces:
                if score < .8:
                    continue
                x_min=int(box[0])
                if x_min < 0:
                    x_min = 0
                y_min=int(box[1])
                if y_min < 0:
                    y_min = 0
                x_max=int(box[2])
                y_max=int(box[3])
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                # Crop image
                img = frame[y_min:y_max, x_min:x_max]
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                img = self.transformations(im_pil)
                img  = Variable(img).cuda(self.gpu)
                img  = img.unsqueeze(0) 
                
                # gaze prediction
                gaze_pitch, gaze_yaw = self.model(img)
                
                pitch_predicted = self.softmax(gaze_pitch)
                yaw_predicted = self.softmax(gaze_yaw)
                
                # Get continuous predictions in degrees.
                pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 4 - 180
                yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 4 - 180
                
                pitch_predicted = pitch_predicted.cpu().detach().numpy()* np.pi/180.0
                yaw_predicted = yaw_predicted.cpu().detach().numpy()* np.pi/180.0
                
                # adult/child 
                if (x_max > child_loc[0] and x_max < child_loc[2] \
                and y_max > child_loc[1] and y_max < child_loc[3]):
                    child_feats['pitch'].append(pitch_predicted)
                    child_feats['yaw'].append(yaw_predicted)
                elif (x_max > adult_loc[0] and x_max < adult_loc[2] \
                and y_max > adult_loc[1] and y_max < adult_loc[3]):
                    adult_feats['pitch'].append(pitch_predicted)
                    adult_feats['yaw'].append(yaw_predicted)
                draw_gaze(x_min,y_min,bbox_width, bbox_height,imgRGB,\
                (pitch_predicted,yaw_predicted),color=(0,0,25))
                cv2.rectangle(imgRGB, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
            # find face landmarks
            lmks = find_face_lmks(frame, imgRGB)
        return imgRGB, adult_feats, child_feats, lmks
    
def find_face_lmks(frame, imgRGB):
    results = face.process(frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
          mp_drawing.draw_landmarks(
              image=imgRGB,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_tesselation_style())
          mp_drawing.draw_landmarks(
              image=imgRGB,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_CONTOURS,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_contours_style())
          mp_drawing.draw_landmarks(
              image=imgRGB,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_iris_connections_style())
        return results.multi_face_landmarks
    return None
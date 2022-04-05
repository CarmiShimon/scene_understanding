import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd

# mediapipe - for hand detection and tracking
import mediapipe as mp
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=4, min_detection_confidence=0.35, min_tracking_confidence=0.2)
mpDraw = mp.solutions.drawing_utils

import torch
import torchvision
from torchvision.transforms import transforms as transforms
from segmentation_utils import draw_segmentation_map, get_outputs
transform = transforms.Compose([transforms.ToTensor()])




def get_hand_features(handLms):
    '''Measure hands' landmarks distance'''
    
    # wrist to middle left
    dist = np.sqrt((handLms.landmark[0].x - handLms.landmark[5].x)**2 + \
    (handLms.landmark[0].y - handLms.landmark[5].y)**2 + (handLms.landmark[0].z - handLms.landmark[5].z)**2) 
    
    return dist

def get_hand_label(index, handLms, results, width, height):
    '''Match hand to classification score and label (left/right)
    Left or right classification isn't accurate'''
    
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            # process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            # extract coordinates
            coords = tuple(np.multiply(np.array((handLms.landmark[mpHands.HandLandmark.WRIST].x, \
            handLms.landmark[mpHands.HandLandmark.WRIST].y)), [width, height]).astype(int))
            output = text, (coords[0], np.max([coords[1]+20, height]))
    return output

# --------------------------------------------- #
# ------------- Load MaskRCNN Model -------------------- #
def load_segmentation_model():
    '''Load a pre-trained segmentation model'''
    # initialize the model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, 
                                                               num_classes=91)
    # set the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the modle on to the computation device and set to eval mode
    model.to(device).eval()
    return model

def get_segments(image, imgRGB, model):
    # transform to convert the image to tensor
    tensor_image = transform(image)
    masks, boxes, labels, scores = get_outputs(tensor_image, model, threshold=0.5)
    imgRGB, found_person_labels = draw_segmentation_map(imgRGB, masks, boxes, labels)
    return imgRGB, masks, boxes, labels, scores, found_person_labels
# --------------------------------------------- #
# ------------- Load YOLO Model -------------------- #
def load_model():
    '''Load a pre-trained ultralytics model'''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = torch.hub.load('ultralytics/yolov5', 'yolo_models/yolov5x6', pretrained=True)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l6', pretrained=True)
    model.to(device).eval()
    return model


def get_dominant_contours(img_shape, contours):
    ''' Find the biggest 2 person contours
    This function require to use person segmentation prior '''
    
    height, width = img_shape
    lengths = [contour.shape[0] for contour in contours]
    idxs = np.argsort(np.asarray(lengths))
    return [contours[idxs[-1]], contours[idxs[-2]]]


def get_person_bbxs(boxes, labels, scores):
    ''' Filter person bboxes from MaskRCNN detector results \
    return exactly 2 bboxes where confidence > 40%, otherwise - None '''
    person_df = pd.DataFrame({'xmin': [], 'ymin': [], 'xmax': [], 'ymax': [], 'confidence': [], 'mask_idx': []})
    person_labels = [i for i, l in enumerate(labels) if l == 'person']
    if person_labels:
        person_scores = [scores[sc] for sc in person_labels]
        if len(person_scores) >= 2:
            max_person_scores = person_scores[:2]
            for l in person_labels:
                if scores[l] in max_person_scores:
                    conf = scores[l]
                    if conf < 0.5:
                        continue
                    xmin, ymin = boxes[l][0]
                    xmax, ymax = boxes[l][1]
                    person_df.loc[-1] = [xmin, ymin, xmax, ymax, conf, l]
                    person_df.index = person_df.index + 1
                    person_df = person_df.sort_index()  # sorting by index
                
    if len(person_df) > 1:
        return person_df
    return None


def get_hands_loc(img, imgRGB):
    ''' Finds hand centers and middle finger tip (x, y) coords '''
    hand_results = hands.process(img)
    height, width, _ = img.shape
    if hand_results.multi_hand_landmarks:
        num_hands = 0
        hand_centers = []
        finger_tips = []
        for num, handLms in enumerate(hand_results.multi_hand_landmarks):
            mpDraw.draw_landmarks(imgRGB, handLms, mpHands.HAND_CONNECTIONS)
            # hand center in pixels
            Cx = int(handLms.landmark[0].x*width)
            Cy = int(handLms.landmark[0].y*height)
            num_hands += 1
            hand_centers.append((Cx, Cy))
            # Finger tip
            Fx = int(handLms.landmark[8].x*width)
            Fy = int(handLms.landmark[8].x*height)
            finger_tips.append((Fx, Fy))
            # get label (in case we need the accuracy and left/right classification)
            # if get_hand_label(num, handLms, hand_results, width, height):
            #   text, coord = get_label(num, handLms, hand_results, width, height)
            #   cv2.putText(img, text, coord, cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
        return hand_centers, finger_tips
    return None, None

def match_rect_to_person(person_df):#, faceBoxes, ages):
    """ Assign bbox to adult/child based on rectangle's diagonal length """
    
    diagonals = []
    for row in person_df.iterrows():
        diagonals.append(np.sqrt((row[1]['xmax'] - row[1]['xmin'])**2 + (row[1]['ymax'] - row[1]['ymin'])**2))
    return np.argmin(diagonals), np.argmax(diagonals) # child, adult


def match_hands_to_person_diag(adult_loc, child_loc, hands_centers, fingers_tips):
    ''' Finds hands inside a person rectangle (child/adult) '''
    
    hands_dict = {'child_hands': [], 'adult_hands': []}
    fingers_dict = {'child_fingers': [], 'adult_fingers': []}
    for i, (hand_center, finger_tip) in enumerate(zip(hands_centers, fingers_tips)):
        # find if a given point lies inside a given rectangle or not
        # find if hand in child rect #
        if (hand_center[0] > child_loc[0] and hand_center[0] < child_loc[2] \
        and hand_center[1] > child_loc[1] and hand_center[1] < child_loc[3]):
            hands_dict['child_hands'].append(hand_center)
            fingers_dict['child_fingers'].append(finger_tip)
        # find if hand in adult rect #
        elif (hand_center[0] > adult_loc[0] and hand_center[0] < adult_loc[2] \
        and hand_center[1] > adult_loc[1] and hand_center[1] < adult_loc[3]):
            hands_dict['adult_hands'].append(hand_center)
            fingers_dict['adult_fingers'].append(finger_tip)
    return hands_dict, fingers_dict


def draw_rect(img, rect, font_size, font_stroke, label):
    ''' draws a rectangle on a given image '''
    cv2.rectangle(img, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (0, 255, 0), font_stroke)
    y = int(rect[1]) - 15 if int(rect[1]) - 15 > 15 else int(rect[1]) + 15
    cv2.putText(img, label, (int(rect[0]), y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_stroke)
    

def get_video_params(vid_filename):
    cap = cv2.VideoCapture(vid_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_bin = 1/fps
    im_size = (int(cap.get(3)), int(cap.get(4))) # width, height
    font_size = np.max([im_size[0]/600, 1.2])
    font_stroke = int(min(im_size[0], im_size[1]) / 130)
    
    return cap, fps, im_size, font_size, font_stroke
    
def img_preprocess(img, vid_ext):
    ''' handle all types of videos, returns up-right RGB image '''
    if vid_ext == 'MOV':
        img = cv2.flip(img, 0)
        img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return imgRGB
    

def collect_and_draw_hands(imgRGB, fingers_dict, hands_dict, child_fingers_seq, adult_fingers_seq, font_size, font_stroke):
    '''collect fingers location of both child and adult into a list'''
    
    child_fingers, adult_fingers = [], []
    
    for child_hand_loc, child_finger_loc in zip(hands_dict['child_hands'], fingers_dict['child_fingers']):
      child_fingers.append(child_finger_loc)
      cv2.putText(imgRGB, 'child_hand', child_hand_loc, cv2.FONT_HERSHEY_PLAIN, font_size, (255, 0, 0), font_stroke, cv2.LINE_AA)
      
    for adult_hand_loc, adult_finger_loc in zip(hands_dict['adult_hands'], fingers_dict['adult_fingers']):
      adult_fingers.append(adult_finger_loc)
      cv2.putText(imgRGB, 'adult_hand', adult_hand_loc, cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 255), font_stroke, cv2.LINE_AA)

    # Equalize number of hands for child and adult 
    max_n_hands = np.max([len(adult_fingers), len(child_fingers)])
    for k in range(max_n_hands):
        if len(adult_fingers) > len(child_fingers):
            child_fingers.append([])
        else:
            adult_fingers.append([])
        
        
    return adult_fingers, child_fingers
    
def match_hands_to_person_polygon(masks, found_person_labels, adult_loc, child_loc, hands_centers, fingers_tips):
    ''' Finds hands inside a person polygon (child/adult) '''
    
    mask_0 = masks[found_person_labels[0]]
    mask_1 = masks[found_person_labels[1]]
    if mask_0.sum() < mask_1.sum():
        child, adult = 'mask_0', 'mask_1'
    else:
        child, adult = 'mask_1', 'mask_0'
        
    hands_dict = {'child_hands': [], 'adult_hands': []}
    fingers_dict = {'child_fingers': [], 'adult_fingers': []}
    
    for i, (hand_center, finger_tip) in enumerate(zip(hands_centers, fingers_tips)):
        # find the distance from hand center to a polygon
        dist_0 = dist_from_contour(mask_0, hand_center, False)
        dist_1 = dist_from_contour(mask_1, hand_center, False)
        if dist_0 > dist_1 and child == 'mask_0':
            hands_dict['child_hands'].append(hand_center)
            fingers_dict['child_fingers'].append(finger_tip)
        # find if hand in adult rect #
        elif dist_0 > dist_1 and child == 'mask_1':
            hands_dict['adult_hands'].append(hand_center)
            fingers_dict['adult_fingers'].append(finger_tip)
        elif dist_1 > dist_0 and child == 'mask_1':
            hands_dict['child_hands'].append(hand_center)
            fingers_dict['child_fingers'].append(finger_tip)
        elif dist_1 > dist_0 and child == 'mask_0':
            hands_dict['adult_hands'].append(hand_center)
            fingers_dict['adult_fingers'].append(finger_tip)
        elif dist_0 == dist_1 and child == 'mask_1':
            if np.abs(dist_from_contour(mask_0, hand_center, True)) < \
                np.abs(dist_from_contour(mask_1, hand_center, True)):
                hands_dict['adult_hands'].append(hand_center)
                fingers_dict['adult_fingers'].append(finger_tip)
            else:
                hands_dict['child_hands'].append(hand_center)
                fingers_dict['child_fingers'].append(finger_tip)
    return hands_dict, fingers_dict
    
def dist_from_contour(mask, hand_center, flag=False):
    mask = np.array(mask * 255, dtype='uint8')
    border = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0 )
    contours, hierarchy = cv2.findContours(border, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))
    return cv2.pointPolygonTest(contours[0], hand_center, flag) # distance
    
def match_face_lmks_to_person(face_lmks, adult_loc, child_loc, img_shape):
    ''' whether face lanmarks belongs to a child or adult rectangle ''' 
    
    child_lmks, adult_lmks = None, None
    height, width, _ = img_shape
    for lmks in face_lmks:
        # check center of face
        if (lmks.landmark[14].x*width > child_loc[0] and lmks.landmark[14].x*width < child_loc[2] \
        and lmks.landmark[14].y*height > child_loc[1] and lmks.landmark[14].y*height < child_loc[3]):
            child_lmks = lmks
        if (lmks.landmark[14].x*width > adult_loc[0] and lmks.landmark[14].x*width < adult_loc[2] \
        and lmks.landmark[14].y*height > adult_loc[1] and lmks.landmark[14].y*height < adult_loc[3]):
            adult_lmks = lmks
    
    return child_lmks, adult_lmks
    
def get_mouth_dist(adult_lmks, child_lmks, img_shape):
    ''' measure distance of several mouth points:\
    (81, 178), (82, 87), (13, 14), (312, 317), (311, 402) '''
    
    height, width, _ = img_shape
    adult_dist, child_dist = None, None
    mouth_points = [(81, 178), (82, 87), (13, 14), (312, 317), (311, 402)]
    if adult_lmks:
        adult_dist = 0
        for p in mouth_points:
            adult_dist += np.sqrt((adult_lmks.landmark[p[0]].x*width - adult_lmks.landmark[p[1]].x*width)**2 +\
                    (adult_lmks.landmark[p[0]].y*height - adult_lmks.landmark[p[1]].y*height)**2)
        adult_dist /= 5 # avg. dist
        
    if child_lmks:
        child_dist = 0
        for p in mouth_points:
            child_dist += np.sqrt((child_lmks.landmark[p[0]].x*width - child_lmks.landmark[p[1]].x*width)**2 +\
                    (child_lmks.landmark[p[0]].y*height - child_lmks.landmark[p[1]].y*height)**2)
        child_dist /= 5 # avg. dist
    return adult_dist, child_dist
    

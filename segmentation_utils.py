import torch
import cv2
import numpy as np
import random
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names


# this will help us create a different color for each class
# COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
COLORS = np.array([[180 , 0, 0],
        [0, 0, 180]])

def get_outputs(image, model, threshold):
    with torch.no_grad():
        # forward pass of the image through the modle
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image = image.unsqueeze(0).to(device)
        outputs = model(image)
    
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    masks = (outputs[0]['masks'] > threshold).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]
    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    # get the classes labels
    labels = [coco_names[i] for i in outputs[0]['labels']]
    return masks, boxes, labels, scores
    
def draw_segmentation_map(image, masks, boxes, labels):
    alpha = 1 
    beta = 0.6 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum

    person_masks_idx = [] # counter for masking only 2 person with highest scores
    xmins = np.asarray([xmin[0][0] for i, xmin in enumerate(boxes) if labels[i]=='person'])
    if len(xmins) >= 2:
        most_left_person_idx = np.argmin(xmins)
        for i in range(len(masks)):
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            green_map = np.zeros_like(masks[i]).astype(np.uint8)
            blue_map = np.zeros_like(masks[i]).astype(np.uint8)
            # apply a random color mask to each object
            # color = COLORS[random.randrange(0, len(COLORS))]
            if i == most_left_person_idx:
                color = COLORS[0]
            else:
                color = COLORS[1]
            red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color
            # combine all the masks into a single image
            segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
            #convert the original PIL image into NumPy format
            image = np.array(image)
            # apply mask on the image
            if labels[i] == 'person': # apply mask only on people
                cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
                person_masks_idx.append(i)
            if len(person_masks_idx) == 2:
                break
        
        # ignore bboxes of large objects
        # if labels[i] in ['couch', 'chair', 'bed', '__background__', 'dining table']:
        #     continue
        # draw the bounding boxes around the objects
        # cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, 
        #               thickness=2)
        # # put the label text above the objects
        # cv2.putText(image , labels[i], (boxes[i][0][0], boxes[i][0][1]-10), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
        #             thickness=2, lineType=cv2.LINE_AA)
    
    return image, person_masks_idx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

def get_accurate_loc_embds(prev_embds, current_embds, adult_prev_idx, child_prev_idx):
    if current_embds.shape[0] < 2 or prev_embds.shape[0] < 2:
        return adult_prev_idx, child_prev_idx
    else:
        sim_ch = 1 - spatial.distance.cosine(prev_embds[child_prev_idx], current_embds[child_prev_idx])
        # sim_ad = 1 - spatial.distance.cosine(prev_embds[adult_prev_idx], current_embds[current_embds])
        sim_ch_ad = 1 - spatial.distance.cosine(prev_embds[child_prev_idx], current_embds[adult_prev_idx])
        if sim_ch > sim_ch_ad:
            return adult_prev_idx, child_prev_idx
        else:
            child_prev_idx, adult_prev_idx


def get_accurate_loc(adult_xmins, child_xmins, person_df, adult_rect_idx, child_rect_idx):
    ''' compute adult and child x location based on time statistics \
    for stable classification results '''
    # ----------- #
    adult_loc_string = 'left'
    if np.mean(np.asarray(adult_xmins)) > np.mean(np.asarray(child_xmins)):
        adult_loc_string = 'right'

    adult_loc = (person_df.iloc[adult_rect_idx]['xmin'], person_df.iloc[adult_rect_idx]['ymin'],\
                person_df.iloc[adult_rect_idx]['xmax'], person_df.iloc[adult_rect_idx]['ymax'])
    child_loc = (person_df.iloc[child_rect_idx]['xmin'], person_df.iloc[child_rect_idx]['ymin'],\
                person_df.iloc[child_rect_idx]['xmax'], person_df.iloc[child_rect_idx]['ymax'])
    
    if ((adult_loc[0] < child_loc[0]) and adult_loc_string == 'right') or\
        ((adult_loc[0] > child_loc[0]) and adult_loc_string == 'left'): 
        # switch child and adult according to statistics
        adult_loc, child_loc = child_loc, adult_loc
        adult_rect_idx, child_rect_idx = child_rect_idx, adult_rect_idx
    return adult_loc, child_loc, adult_rect_idx, child_rect_idx


class HandInteraction:
    def __init__(self, fps, width, height):
        self.fps = fps
        self.width = width
        self.height = height
        self.pixels_min_dist = width/50 # was 40
        self.min_finger_samples = int(fps/4) # finger samples - at least 0.25 sec
        self.min_interaction_time = int(fps/8) # must be integer = #frames
        
    def hands_proximity(self, child_wrists, adult_wrists):
        ''' Find proximity of finger tips in sequence of frames '''
        # child_finger_tips = np.asarray(child_finger_tips)
        # adult_finger_tips = np.asarray(adult_finger_tips)
        # if np.any(child_finger_tips) and np.any(child_finger_tips):
            # child_finger_tips = child_finger_tips[child_finger_tips != None] #[l for l in child_fingers_seq if l]
            # adult_finger_tips = adult_finger_tips[adult_finger_tips != None] #[k for k in adult_fingers_seq if k]
            # make sure there are at least 0.25 sec finger samples
            # if len(child_finger_tips) > self.min_finger_samples and len(adult_finger_tips) > self.min_finger_samples: 
        pixel_dists = []
        for child_wrist, adult_wrist in zip(child_wrists, adult_wrists):
            if child_wrist and adult_wrist:
                pixel_dists.append(np.sqrt((child_wrist[0][0] - adult_wrist[0][0])**2 + (child_wrist[0][1] - adult_wrist[0][1])**2))
                pixel_dists.append(np.sqrt((child_wrist[1][0] - adult_wrist[1][0]) ** 2 + (child_wrist[1][1] - adult_wrist[1][1]) ** 2))
        pixel_dists = np.asarray(pixel_dists)
        # hands are close enough for at least 0.125 sec
        if len(pixel_dists[pixel_dists < self.pixels_min_dist]) > self.min_interaction_time: 
            return True
    
        return None
        
        
def get_speaker_by_mouth(adult_mouth_seq, child_mouth_seq):
    ''' estimate speaker by mouth movements over time '''
    is_adult_speaks = False
    is_child_speaks = False
    
    adult_seq = np.asarray([d for d in adult_mouth_seq if d is not None])
    child_seq = np.asarray([d for d in child_mouth_seq if d is not None])
    
    adult_var = np.var(adult_seq)
    child_var = np.var(child_seq)
    if adult_var > 3 and len(adult_seq) > 4:
        is_adult_speaks = True
    if child_var > 3 and len(adult_seq) > 4:
        is_child_speaks = True
    return is_adult_speaks, is_child_speaks
        
def gaze_to_face(pred_phases, true_phase, time_len, fps):
    ''' check if a child is looking at adult's face '''
    is_contact = False
    degrees_thd = 14
    min_samples = int(fps*time_len/2)
    true_phase = np.asarray(true_phase) 
    avg_true_phase = np.mean(true_phase[true_phase != None]) # mean abs phase array
    
    pred_phases = np.asarray(pred_phases)
    preds = pred_phases[pred_phases != None]
    # preds[(preds >= -180) & (preds <= -176)] = 180 # almost zero angle
    
    min_attraction_time = int(len(preds)*0.25) # num of frames
    if len(preds) > min_samples:
        dists = preds - avg_true_phase
        if len(dists[(dists >= -degrees_thd) & (dists <= degrees_thd)]) > min_attraction_time:
            is_contact = True
    return is_contact
    
    
def gaze_to_adult_hands(pred_phases, child_face_seq, adult_hands_seq, time_len, fps):
    ''' check if a child is looking at adult's hands '''
    is_contact = False
    degrees_thd = 14
    min_samples = int(fps*time_len/2)
    # pred_phases = np.asarray(pred_phases)
    # preds = pred_phases[pred_phases != None]
    min_attraction_time = int(len(pred_phases)*0.25) # num of frames
    # preds[(preds >= -180) & (preds <= -176)] = 180 # almost zero angle
    # 2 hands handling
    N = 2
    adult_2_hands_seq = [adult_hands_seq[n:n+N] for n in range(0, len(adult_hands_seq), N)]
    # abs_phases = []
    dists = []
    for child_face, adult_hands, pred_phase in zip(child_face_seq, adult_2_hands_seq, pred_phases):
        if child_face and pred_phase:
            for adult_hand in adult_hands[0]:
                if adult_hand:
                    a, b, c, d = child_face # x_min,y_min,bbox_width, bbox_height
                    child_center_of_face = (int(a+c / 2.0), int(b+d / 2.0)) # center of face location
                    dy_face_hand = child_center_of_face[1] - adult_hand[1]
                    dx_face_hand = child_center_of_face[0] - adult_hand[0]
                    ch_ad_face_hand_phase = np.arctan2(dy_face_hand, dx_face_hand) * 180 / np.pi
                    if (pred_phase >= -180) and (pred_phase <= -176):
                        pred_phase = 180
                    dists.append(pred_phase - ch_ad_face_hand_phase)
                    # abs_phases.append(ch_ad_face_hand_phase)
        
    if len(dists) >= min_samples:
        dists = np.asarray(dists)
        # min_len = min([len(preds), len(abs_phases)])
        # dists = preds[:min_len] - np.asarray(abs_phases[:min_len])
        if len(dists[(dists >= -degrees_thd) & (dists <= degrees_thd)]) > min_attraction_time:
            is_contact = True
    return is_contact
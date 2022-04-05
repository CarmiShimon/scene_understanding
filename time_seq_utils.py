import numpy as np


def get_accurate_loc(adult_xmins, child_xmins, person_df, adult_rect_idx, child_rect_idx):
    ''' compute adult and child x location based on time statistics \
    for stable classification results '''
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
    return adult_loc, child_loc


class HandInteraction:
    def __init__(self, fps, width, height):
        self.fps = fps
        self.width = width
        self.height = height
        self.pixels_min_dist = width/40
        self.min_finger_samples = int(fps/4) # finger samples - at least 0.25 sec
        self.min_interaction_time = int(fps/8) # must be integer = #frames
        
    def hands_proximity(self, child_fingers_seq, adult_fingers_seq):
        ''' Find proximity of finger tips in sequence of frames '''
        
        if child_fingers_seq and adult_fingers_seq:
            child_finger_tips = [l for l in child_fingers_seq if l]
            adult_finger_tips = [k for k in adult_fingers_seq if k]
            # make sure there are at least 0.25 sec finger samples
            if len(child_finger_tips) > self.min_finger_samples and len(adult_finger_tips) > self.min_finger_samples: 
                pixel_dists = []
                for child_finger, adult_finger in zip(child_fingers_seq, adult_fingers_seq):
                    if child_finger and adult_finger:
                        pixel_dists.append(np.sqrt((child_finger[0] - adult_finger[0])**2 + (child_finger[1] - adult_finger[1])**2))
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
        
        
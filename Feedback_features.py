import pandas as pd
import numpy as np
from ast import literal_eval
import math


class Features:
    def __init__(self, win_len=0.5, overlap_prc=0.5, angle_delta=20, min_interaction_time=0.2):
        self.win_len = win_len  # sec
        self.hands_thd = 0.2  # sec
        self.overlap_prc = overlap_prc  # sec
        self.spk_thd = 0.2  # sec = 200 msec
        self.angle_delta = angle_delta  # degrees
        self.gaze_thd = 0.15  # 0.4 second is enough for gaze direction prediction (Keren talked about glimpse from child)

        self.win_len_frames = 0  # num frames
        self.overlap_len_frames = 0  # num frames
        self.spk_frames_thd = 0
        self.gaze_frames_thd = 0
        self.hands_frames_thd = 0
        self.min_interaction_time = min_interaction_time
        self.min_interaction_frames = 3
        # self.max_width = 100
        self.play_actions = ['carrying baby', 'shaking hands', 'tango dancing', 'balloon blowing', 'laughing', 'hugging']
        self.pause_actions = []

    # --------------------------------------------------------- #
    def get_speaker(self, win_df):
        speaker = None
        speaker_count = win_df['speaker'].value_counts()
        if np.any(speaker_count.values):
            if speaker_count.values[0] > self.spk_frames_thd:
                speaker = speaker_count.index[0]
            # if speaker == 'b':
            #     speaker = 'c'
        return speaker

    # --------------------------------------------------------- #
    def get_child_gaze_direction(self, win_df):
        is_face_looking, is_hand_looking = False, False
        # use values only when both angle are not nan, also when we have adult hands and child face
        win_df_gaze_face = win_df.dropna(subset=['pred_phase', 'abs_phase'], how='any')  # drop nans
        if len(win_df_gaze_face) < self.gaze_frames_thd:
            is_face_looking = False
        else:
            face_looking_frames = np.sum(np.abs(win_df_gaze_face['pred_phase'].values - win_df_gaze_face['abs_phase'].values) <= self.angle_delta)
            # print(face_looking_frames)
            if face_looking_frames >= self.gaze_frames_thd:
                is_face_looking = True

        win_df_gaze_hands = win_df.dropna(subset=['pred_phase', 'adult_hands_abs_phase'], how='any')  # drop nans
        if len(win_df_gaze_hands) < self.gaze_frames_thd:
            return is_face_looking, is_hand_looking
        else:
            hand_looking_frames = np.sum(np.abs(win_df_gaze_face['pred_phase'].values - win_df_gaze_face['adult_hands_abs_phase'].values) <= self.angle_delta)
            if hand_looking_frames >= self.gaze_frames_thd:
                is_hand_looking = True

        return is_face_looking, is_hand_looking

    # --------------------------------------------------------- #
    def find_abs_phase_hands(self, win_df_gaze):
        abs_phases = []
        if len(win_df_gaze) < self.gaze_frames_thd:
            return abs_phases
        for idx, row in win_df_gaze.iterrows():
            try:
                # ---- converting text to array ---- #
                # child_loc = literal_eval(row['child_loc'].replace(' ', ',')) if row['child_loc'][0] == '[' else literal_eval(row['child_loc'])
                child_face_x, child_face_y = literal_eval(row['center_of_child_face'])
                # a, b, c, d = literal_eval(child_locs[0]), literal_eval(child_locs[1]), literal_eval(child_locs[2]), literal_eval(child_locs[3])
                # a, b, c, d = child_loc  # x_min,y_min,x_max,y_max
                # child_center_of_face = (int(a + c / 2.0), int(b + d / 2.0))  # center of face location
                hands_phases = []
                # --- go over all adult hands --> find the minimal angle (child gaze to adult hand) --- #
                if type(row['adult_hands_loc']) is str:
                    for hand in literal_eval(row['adult_hands_loc'])[0]:
                      dy_face_hand = hand[1] - child_face_y
                      dx_face_hand = hand[0] - child_face_x
                      ch_ad_face_hand_abs_phase = np.arctan2(dy_face_hand, np.abs(dx_face_hand)) * 180 / np.pi
                      if child_face_y < hand[1]:
                          ch_ad_face_hand_abs_phase = -np.abs(ch_ad_face_hand_abs_phase)
                      else:
                          ch_ad_face_hand_abs_phase = np.abs(ch_ad_face_hand_abs_phase)
                      hands_phases.append(ch_ad_face_hand_abs_phase)
                    abs_phases.append(np.min(np.asarray(hands_phases)))  # get min gaze phase to hand (dealing with multiple hands)
                elif math.isnan(row['adult_hands_loc']):
                    abs_phases.append(np.nan)
            except:
                abs_phases.append(np.nan)
        return abs_phases

    # --------------------------------------------------------- #
    def get_hands_proximity(self, win_df, hands_proximity_value):
        is_hands_getting_closer = False
        win_df_hands = win_df.dropna(subset=['adult_hands_loc', 'child_hands_loc'], how='any')  # drop nans
        if len(win_df_hands) < self.hands_frames_thd:
            return is_hands_getting_closer
        hands_proximity = []
        for idx, row in win_df_hands.iterrows():
            proximities = []
            for adult_hand in literal_eval(row['adult_hands_loc'])[0]:
                for child_hand in literal_eval(row['child_hands_loc'])[0]:
                    # hands euclidean distance
                    # self.max_width = np.max([child_hand[0], adult_hand[0]]) if np.max([child_hand[0], adult_hand[0]]) > self.max_width else self.max_width
                    proximities.append(np.sqrt(np.square(adult_hand[0] - child_hand[0]) + np.square(adult_hand[1] - child_hand[1])))
            # Find minimum hands distance proximity (out of all hands)
            hands_proximity.append(np.min(np.asarray(proximities)))

        # Find if hands are getting closer by time
        hands_proximity_arr = np.asarray(hands_proximity)
        if len(hands_proximity_arr) >= self.hands_frames_thd:
            # if (np.sum(np.diff(hands_proximity_arr)) < -25) or (len(hands_proximity_arr[hands_proximity_arr < self.max_width/4]) > self.min_interaction_frames):
            if len(hands_proximity_arr[hands_proximity_arr < hands_proximity_value]) >= self.min_interaction_frames:
                is_hands_getting_closer = True
        return is_hands_getting_closer

    # --------------------------------------------------------- #
    def get_hands_dist(self, win_df):
        is_similar_dist = False
        win_df_hands = win_df.dropna(subset=['adult_hands_loc', 'child_hands_loc'], how='any')  # drop nans
        if len(win_df_hands) < 8:
            return is_similar_dist
        right_hands_proximity = []
        left_hands_proximity = []
        for idx, row in win_df_hands.iterrows():
            # hands euclidean distance
            # self.max_width = np.max([child_hand[0], adult_hand[0]]) if np.max([child_hand[0], adult_hand[0]]) > self.max_width else self.max_width
            ad_right_hand = literal_eval(row['adult_hands_loc'])[0][0]
            ad_left_hand = literal_eval(row['adult_hands_loc'])[0][1]
            ch_right_hand = literal_eval(row['child_hands_loc'])[0][0]
            ch_left_hand = literal_eval(row['child_hands_loc'])[0][0]
            right_hands_proximity.append(np.sqrt(np.square(ad_right_hand[0] - ch_right_hand[0]) + np.square(ad_right_hand[1] - ch_right_hand[1])))
            left_hands_proximity.append(np.sqrt(
                np.square(ad_left_hand[0] - ch_left_hand[0]) + np.square(ad_left_hand[1] - ch_left_hand[1])))

        hands_dist = np.asarray(left_hands_proximity) - np.asarray(right_hands_proximity)
        if len(hands_dist[hands_dist < 15]) > 8:
            is_similar_dist = True
        return is_similar_dist

    # --------------------------------------------------------- #
    def get_action(self, win_df):
        action = None
        if win_df['action'].any():
            df = win_df[win_df.select_dtypes(float).notna().any(axis=1)]
            action_idx = np.where(df['action'].notna())[0][0]
            actions = literal_eval(df.iloc[action_idx]['action'])
            for action in actions:
                if action[0] in self.play_actions:
                    return f'play_{action[0]}'
                if action[0] in self.pause_actions:
                    return f'pause_{action[0]}'
        return np.NaN

    # --------------------------------------------------------- #
    def find_features(self, csv_file, width, is_overlap, fps):

        hands_proximity_thd = 1.8
        vid_df = pd.read_csv(csv_file, index_col=0)
        # find how many frames contains both child and adult
        two_people_df = vid_df.dropna(subset=['center_of_child_face', 'adult_loc'], how='any')

        # find child body width
        child_width = []
        child_locs = vid_df['child_loc'].dropna()
        for idx, row in child_locs.items():
            try:
                row = row[1:-1].split()
                # child_loc = literal_eval(row.replace('  ', ',')) if row[0] == '[' else literal_eval(row)
                # child_loc = literal_eval(row[0])
                a, b, c, d = literal_eval(row[0]), literal_eval(row[1]), literal_eval(row[2]), literal_eval(row[3])
                child_width.append(int(c - a))  # append widths
            except:
                pass

        avg_child_width = np.asarray(child_width).mean()
        hands_proximity_value = int(avg_child_width / hands_proximity_thd)
        time_bin = vid_df.iloc[1]['time']
        vid_fps = fps
        # avg_adult_bbx = vid_df['adult_loc']
        self.win_len_frames = int(self.win_len / time_bin)  # N frames to process
        self.overlap_len_frames = int(self.overlap_prc * self.win_len_frames)  # N frames to overlap

        self.min_interaction_frames = int(self.min_interaction_time * vid_fps)  # sec to frames
        self.hands_frames_thd = int(self.hands_thd * vid_fps)  # sec to frames
        self.spk_frames_thd = int(self.spk_thd * vid_fps)  # sec to frames
        self.gaze_frames_thd = int(self.gaze_thd * vid_fps)  # sec to frames

        frame_idx = 0
        i = 0
        start_time = 0
        video_audio_results_df = pd.DataFrame({'start_time': [], 'speaker': [], 'hand_looking': [], 'face_looking': [],
                                               'hand_interaction': [], 'action': []})  # , 'three_ppl': []})

        while frame_idx + self.win_len_frames < len(vid_df):
            win_df = vid_df.iloc[frame_idx: frame_idx + self.win_len_frames]
            if is_overlap:
                frame_idx += self.overlap_len_frames
            start_time = win_df['time'].values[0]
            frame_idx += self.win_len_frames
            # if win_df['more_than_2_ppl'].sum() > (len(win_df)/2):  # check for more than 2 people
            # is_three_ppl = True
            action = self.get_action(win_df)
            speaker = self.get_speaker(win_df)
            # print('spk: ', speaker)
            is_face, is_hand = self.get_child_gaze_direction(win_df)
            # print('time: ', start_time)
            is_hands_closer = self.get_hands_proximity(win_df, hands_proximity_value)
            # print('is_hands_closer: ', is_hands_closer)
            is_similar_hands_dist = self.get_hands_dist(win_df)
            video_audio_results_df.loc[i] = [start_time, speaker, is_hand, is_face, is_hands_closer, action]  # , is_three_ppl]
            i += 1
        video_audio_results_df['2_ppl_seconds'] = len(two_people_df) / fps
        return video_audio_results_df

    # ----- # ----- # ----- # ----- # ----- # ----- # ----- # ----- # ----- # ----- # ----- # ----- #
    def find_hizuk(self, df, time_res):
        time_factor = int(1 / time_res)
        hand_inter_max_time = 6
        hand_inter_rows = time_factor * hand_inter_max_time
        reinforcement = np.zeros(len(df))
        communicative_speech = np.zeros(len(df))

        time_bin = self.win_len
        for idx, row in df.iterrows():
            if str(row['speaker']) == 'b' or row['speaker'] == 'c':  # child speaks
                if df.iloc[idx - (1 * time_factor):idx + (1 * time_factor)]['face_looking'].any() or \
                      df.iloc[idx - (1 * time_factor):idx + (1 * time_factor)]['hand_looking'].any():
                    communicative_speech[idx:idx + (1 * time_factor)] = 1
                    print('communicative_speech')
                    if df.iloc[idx + 1:idx + hand_inter_rows]['hand_interaction'].any():
                        print('reinforcement')
                        hand_inter_idx = idx + df.iloc[idx:idx + hand_inter_rows]['hand_interaction'].to_list().index(1)  # find time of hand interaction
                        reinforcement[hand_inter_idx:hand_inter_idx + (3 * time_factor)] = 1
                    else:
                        print('no-reinforcement')
                        reinforcement[idx + 3:idx + 3 + (3 * time_factor)] = 2  # no hand interaction
                        if 'a' in df.iloc[idx + 1:idx + 1 + (3 * time_factor)]['speaker'].to_list():  # if adult speaks = double demand
                            reinforcement[idx + 3:idx + 3 + (3 * time_factor)] = 3

                # elif df.iloc[idx:idx+hand_inter_rows]['hand_interaction'].any():
                #   # child speaks + hand interaction
                #   hand_inter_idx = idx + df.iloc[idx:idx+hand_inter_rows]['hand_interaction'].to_list().index(1)
                #   reinforcement[hand_inter_idx:hand_inter_idx + (2*time_factor)] = 1

        return reinforcement, communicative_speech

    def smooth_reinforcement(self, reinforcement):
        reinforcement[-1] = 0
        i = 0
        while i < len(reinforcement) - 1:
            if i == 0 or reinforcement[i] == 0:
                i += 1
                continue
            elif reinforcement[i] == 1:  # reinforcement
                end_rein_idx = np.where(reinforcement[i:] < 1)[0][0]
                reinforcement[i:i + end_rein_idx] = 1
            i += 1
        reinforcement[-1] = 0
        return reinforcement


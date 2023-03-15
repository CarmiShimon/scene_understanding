import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import sys
import ffmpeg
import img_utils
import time_seq_utils
import matplotlib
import math




# -- To install mmdet - install visual studio & visual studio build toold & restart computer -- #
# then - follow instructions in https://mmdetection.readthedocs.io/en/v2.2.0/install.html #
sys.path.append("./gaze_direction")
sys.path.append("./gaze_direction/face-detection")
sys.path.append("./gaze_direction/face-detection/face_detection")
from gaze_direction.gaze import Gaze
from face_detection import RetinaFace
face_detector = RetinaFace(gpu_id=0)


# new gaze
# import torch
# sys.path.append('./dynamic-3d-gaze-from-afar')
# from models.gazenet import GazeNet
# from dataloader.gafa import create_gafa_dataset
# model = GazeNet(n_frames=7)
# model.load_state_dict(torch.load('./dynamic-3d-gaze-from-afar/models/weights/gazenet_GAFA.pth', map_location='cpu')['state_dict'])
# model.cuda()
# sequence = ['living_room/1']
# dataset = create_gafa_dataset(n_frames=7, exp_names=sequence, root_dir='./dynamic-3d-gaze-from-afar/data/preprocessed/')
# resnet
# from torchvision.models import resnet50, ResNet50_Weights
# pose and person detectors
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
# from mmdet.apis import inference_detector, init_detector
pose_config = './mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
pose_checkpoint = './mmpose/checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
det_config = './mmdetection/configs/faster_rcnn_r50_fpn_1x_coco.py'
detector_checkpoint = './mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# ------------------ mmaction2 recognizer ---------------- #
from mmaction2.mmaction.apis import inference_recognizer, init_recognizer
action_config = 'mmaction2/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
action_checkpoint = 'mmaction2/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'
action_labels_file = 'mmaction2/tools/data/kinetics/label_map_k400.txt'
action_labels = open(action_labels_file).readlines()
action_labels = [x.strip() for x in action_labels]
frames_dir = './tmp_frames'
os.makedirs(frames_dir, exist_ok=True)
# -------------------------------------------------- #
plt.interactive(False)
matplotlib.use('TkAgg')

if __name__ == '__main__':
    sub_dir = 'Child 101'  # 'play_pause'
    input_dir = fr'./videos/orig_videos/{sub_dir}/'
    out_dir = fr'./videos/scene_understanding_videos/{sub_dir}'

    os.makedirs(out_dir, exist_ok=True)
    # ---------- constants ------------ #
    T_STATS_RECT = 1  # second to collect information regarding adult or child location
    T_STATS_HANDS = 1  # second to collect information regarding hands closeness
    T_STATS_MOUTH = 0.5  # second to collect information regarding mouth movement
    T_STATS_EYE_CONTACT = 1  # second to collect information regarding eye contact
    T_STATS_ACTION_RECOGNITION = 0.5  # second to run action recognition model
    # load DEEP LEARNING MODELS
    yolo_model = img_utils.load_model()
    gaze_model = Gaze()
    segmentation_model = img_utils.load_segmentation_model()
    action_model = init_recognizer(action_config, action_checkpoint, device='cuda:0')


    video_types = [f'{input_dir}*.MOV', f'{input_dir}*.mp4', f'{input_dir}*.mkv', f'{input_dir}*.avi']
    videos_grabbed = list(itertools.chain(*[glob.glob(v) for v in video_types]))
    for video in videos_grabbed:
        video = video.replace('\\', '/')
        print(video)
        # if not video[-12:] == 'yt_video.mp4':
        if not video.split('/')[-1] == '2.mov' or video.split('/')[-1] == '5.mov' or video.split('/')[-1] == '6a.mov':
            continue
        # if not (video.split('/')[-1] == '6.mp4' or video.split('/')[-1] == '8.mp4'):
        #     continue
        print('processing video: ', video)
        if not os.path.isfile(video):
            continue
        try:
            if cap.isOpened():
                cap.release()
        except:
            print("capture is closed")

        # extract audio from video
        # audio = mp.VideoFileClip(video)
        # break
        # ----------video params---------------- #
        cap, fps, im_size, font_size, font_stroke = img_utils.get_video_params(video)
        print('FPS = ', fps)
        time_bin = 1 / fps
        width, height = im_size[0], im_size[1]
        out_name = f"{out_dir}/{video.split('/')[-1][:-4]}_AI.mp4"
        output_video = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'MP4V'), fps, im_size)
        # -------------------------- #

        # ----------- variables and class instantiation --------- #
        i, timer = 0, 0
        out_df = pd.DataFrame(
            {'time': [], 'adult_loc': [], 'child_loc': [], 'center_of_child_face': [], 'adult_hands_loc': [], 'child_hands_loc': [],
             'adult_mouth_dist': [], 'child_mouth_dist': [], 'pred_phase': [], 'abs_phase': [], 'adult_hands_abs_phase': [],
             'action': [], 'more_than_2_ppl': []})

        adult_xmins_seq, child_xmins_seq, child_hands_seq, adult_hands_seq, child_fingers_seq, adult_fingers_seq = [], [], [], [], [], []
        child_mouth_seq, adult_mouth_seq = [], []
        more_than_2_ppl_seq, action = [], []
        pred_phase_eyes_seq, abs_phase_eyes_seq, pred_phase_eyes_hands_seq, abs_phase_eyes_hands_seq, child_face_seq = [], [], [], [], []
        prev_frame = False
        # interaction #
        is_hands_interaction = None
        is_eye_contact = None
        is_eye_hands_contact = None
        k = 1
        # ----------- variables and class instantiation --------- #
        HandInteraction = time_seq_utils.HandInteraction(fps, width, height)
        pose_model = init_pose_model(pose_config, pose_checkpoint)

        # ---------iterate over frames--------- #
        while (cap.isOpened()):
            success, img = cap.read()
            if not success:
                print("No more frames to process.")
                # If loading a video, use 'break' instead of 'continue'.
                break
            adult_loc, child_loc, adult_hands_loc, child_hands_loc, adult_mouth_dist, child_mouth_dist, pred_phase, abs_phase = None, None, None, None, None, None, None, None
            center_of_child_face, adult_hands_abs_phase = None, None
            imgRGB = img_utils.img_preprocess(img, vid_ext=video[-3:])
            # imgRGB = cv2.flip(imgRGB, 0)
            orig_img_rgb = imgRGB.copy()

            # if i> 5 and i < 7250:
            #     i+=1
            #     continue
            # ----------- 2nd frame ----------- #
            if (i > 0) and (i % 2 == 0) and (out_df.iloc[0]['child_loc'] is not None):  # duplicate results row only when there are child results in the last row
                tmp_arr = out_df.iloc[0].values
                tmp_arr[0] = timer
                out_df.loc[-1] = tmp_arr
                out_df.index = out_df.index + 1
                out_df = out_df.sort_index()  # sorting by index
                timer += time_bin
                img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
                output_video.write(img)
                i += 1
                continue

            # instance segmentation (only for person), bboxes for all objects
            # imgRGB, masks, boxes, labels, scores = img_utils.get_segments(orig_img_rgb, imgRGB, segmentation_model, thd=0.7)

            detector_results = yolo_model(orig_img_rgb)
            boxes = detector_results.pandas().xyxy[0]
            person_df = boxes[(boxes['name'] == 'person') & (boxes['confidence'] > 0.6)][['xmin', 'ymin', 'xmax', 'ymax']]
            # person_df = img_utils.get_person_bbxs((width, height), boxes, labels, scores, min_person_score=0.74)  # return None or at least 2 bboxes
            person_df = img_utils.remove_reflection_bboxes(orig_img_rgb, person_df)
            # remove duplicated person
            # person_df = img_utils.remove_same_detections(person_df)
            more_than_2_ppl = False if person_df.shape[0] <= 2 else True
            # find adult and child based on skeleton length
            if person_df.shape[0] >= 2:
                # ------------ face-based age estimation------------ #
                adult_rect_idx, child_rect_idx, adult_loc, child_loc = img_utils.find_child(orig_img_rgb, face_detector, person_df)
                if (child_rect_idx is None) and (len(child_xmins_seq) > 20):
                    mean_child_x = np.mean(np.asarray(child_xmins_seq)[-18:])
                    mean_adult_x = np.mean(np.asarray(adult_xmins_seq)[-18:])
                    child_rect_idx = np.argmin(np.abs(person_df['xmin'].values - mean_child_x))
                    adult_rect_idx = np.argmin(np.abs(person_df['xmin'].values - mean_adult_x))
                    if child_rect_idx == adult_rect_idx:
                        child_rect_idx, adult_rect_idx = np.argmin(person_df['ymax'].values - person_df['ymin'].values), np.argmax(person_df['ymax'].values - person_df['ymin'].values)
                    adult_loc, child_loc = person_df.iloc[adult_rect_idx].values, person_df.iloc[child_rect_idx].values

            if person_df.shape[0] >= 2 and (child_rect_idx is not None):  # at least 2 person found
                # child_rect_idx, adult_rect_idx = match_rect_to_person(person_df, faceBoxes, ages)
                # -------------------------------------------------- #
                # find person skeleton = inference pose
                person_boxes = [{'bbox': person_df.loc[k].values} for k in range(person_df.shape[0])]
                pose_results, returned_outputs = inference_top_down_pose_model(pose_model, orig_img_rgb, person_boxes,
                                                                               bbox_thr=None, format='xyxy',
                                                                               dataset=pose_model.cfg.data.test.type)
                # person_df, pose_results = img_utils.remove_duplicated_person(person_df, pose_results)
                # more_than_2_ppl = False if person_df.shape[0] <= 2 else True
                if person_df.shape[0] >= 2 and len(pose_results) >= 2:  # at least 2 person found
                    imgRGB = vis_pose_result(pose_model, imgRGB, pose_results, dataset=pose_model.cfg.data.test.type, show=False)

                    # find adult and child based on diagonal length
                    # child_rect_idx, adult_rect_idx = img_utils.match_rect_to_person(person_df)
                    adult_loc, child_loc = person_df.iloc[adult_rect_idx].values[:4], person_df.iloc[child_rect_idx].values[:4]
                    adult_xmins_seq.append(int(adult_loc[0]))
                    child_xmins_seq.append(int(child_loc[0]))


                    # left_hand = pose_results[0]['keypoints'][7]
                    # right_hand = pose_results[0]['keypoints'][4]
                    if prev_frame:
                        prev_frame = False
                        current_embds = img_utils.get_person_embds(orig_img_rgb, person_df)
                        adult_idx, child_idx = time_seq_utils.get_accurate_loc_embds(embds, current_embds, adult_idx, child_idx)
                        adult_loc, child_loc = person_df.iloc[adult_idx], person_df.iloc[child_idx]
                    if (i-1 % 8 == 0) and len(child_xmins_seq) > 5:
                        embds = img_utils.get_person_embds(orig_img_rgb, person_df)
                        adult_loc_prev, child_loc_prev, adult_idx, child_idx = time_seq_utils.get_accurate_loc(adult_xmins_seq, child_xmins_seq, person_df, adult_rect_idx, child_rect_idx)
                        prev_frame = True
                    if child_rect_idx == adult_rect_idx:
                        child_rect_idx, adult_rect_idx = np.argmin(
                            person_df['ymax'].values - person_df['ymin'].values), np.argmax(
                            person_df['ymax'].values - person_df['ymin'].values)

                    if adult_rect_idx >= len(pose_results) or child_rect_idx >= len(pose_results):
                        adult_rect_idx, child_rect_idx = 0, 1

                        # ----------------------------------------------------------------------------- #
                    # if i > int(T_STATS_RECT * fps) and len(adult_xmins_seq) > fps / 3 and len(
                    #         child_xmins_seq) > fps / 3:  # statistics runs for every T_STATS sec
                    #     adult_loc, child_loc = time_seq_utils.get_accurate_loc(adult_xmins_seq[-int(fps / 2):],
                    #                                                            child_xmins_seq[-int(fps / 2):], person_df,
                    #                                                            adult_rect_idx, child_rect_idx)
                    # adult_xmins_seq, child_xmins_seq = [], []
                    # draw the bounding box and label on the image
                    img_utils.draw_rect(imgRGB, adult_loc, font_size, font_stroke, label='adult')
                    img_utils.draw_rect(imgRGB, child_loc, font_size, font_stroke, label='child')

                    # --------------- find gaze and face landmraks ------------- #

                    # pred_phase = (np.arctan2(abs(y1 - y0), abs(x1 - x0)) * 180 / np.pi) * abs((adult_center_of_face_y - child_center_of_face_y)/(y1-y0))
                    # print('pred_phase', pred_phase)
                    # cv2.arrowedLine(imgRGB, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 0), 2)
                    adult_phase_dy = pose_results[adult_rect_idx]['keypoints'][0][1] - pose_results[adult_rect_idx]['keypoints'][4][1]
                    adult_phase_dx = pose_results[adult_rect_idx]['keypoints'][0][0] - \
                                     pose_results[adult_rect_idx]['keypoints'][4][0]
                    child_phase_dy = pose_results[child_rect_idx]['keypoints'][0][1] - \
                                     pose_results[child_rect_idx]['keypoints'][4][1]
                    child_phase_dx = pose_results[child_rect_idx]['keypoints'][0][0] - \
                                     pose_results[child_rect_idx]['keypoints'][4][0]
                    adult_pred_phase = math.atan2(adult_phase_dy, adult_phase_dx) * 180 / np.pi
                    child_pred_phase = math.atan2(child_phase_dy, child_phase_dx) * 180 / np.pi
                    imgRGB, pred_phase = gaze_model.find_gaze(orig_img_rgb, imgRGB, adult_loc, child_loc, adult_pred_phase, child_pred_phase)
                    if pred_phase:
                        x0 = pose_results[child_rect_idx]['keypoints'][4][0]  # head
                        child_face_x = pose_results[child_rect_idx]['keypoints'][1][0]  # Eye
                        y0 = pose_results[child_rect_idx]['keypoints'][4][1]
                        child_face_y = pose_results[child_rect_idx]['keypoints'][1][1]
                        child_center_of_face_x, child_center_of_face_y = pose_results[child_rect_idx]['keypoints'][1][
                                                                             0], \
                                                                         pose_results[child_rect_idx]['keypoints'][1][
                                                                             1]  # nose
                        adult_center_of_face_x, adult_center_of_face_y = pose_results[adult_rect_idx]['keypoints'][1][
                                                                             0], \
                                                                         pose_results[adult_rect_idx]['keypoints'][1][1]
                        abs_phase = np.arctan2(child_face_y - adult_center_of_face_y, child_face_x - adult_center_of_face_x) * 180 / np.pi
                        center_of_child_face = (child_center_of_face_x, child_center_of_face_y)
                        if pred_phase < 0:
                            pred_phase += 360
                        if abs_phase < 0:
                            abs_phase + 360
                    else:
                        pred_phase, abs_phase, center_of_child_face = None, None, None
                    # pred_phase_eyes_seq.append(pred_phase), abs_phase_eyes_seq.append(abs_phase), child_face_seq.append(
                    #     child_face)
                    # # print phases
                    # if np.any(pred_phase) and np.any(abs_phase):
                    #     cv2.putText(imgRGB, 'pred:' + str(round(pred_phase, 2)),
                    #                 (int(width / 2) + 30, int(height / 2) + 30), cv2.FONT_HERSHEY_PLAIN, font_size,
                    #                 (0, 255, 0), 5, cv2.LINE_AA)
                    #     cv2.putText(imgRGB, 'abs:' + str(round(abs_phase, 2)), (int(width / 2), int(height / 2)),
                    #                 cv2.FONT_HERSHEY_PLAIN, font_size, (255, 0, 0), 5, cv2.LINE_AA)
                    # # --------------- Find child gaze direction from phase estimation ------------- #
                    # if i % int(T_STATS_EYE_CONTACT * fps) == 0 and i > 0:  # statistics runs for every T_STATS_EYE_CONTACT sec
                    #     is_eye_contact = False
                    #     is_eye_hands_contact = False
                    #     is_eye_contact = time_seq_utils.gaze_to_face(pred_phase_eyes_seq[-int(T_STATS_EYE_CONTACT * fps):],
                    #                                                  abs_phase_eyes_seq[-int(T_STATS_EYE_CONTACT * fps):],
                    #                                                  T_STATS_EYE_CONTACT, fps)
                    #     # adult_hands_seq[-int(T_STATS_EYE_CONTACT*fps)*2:] # since we have 2 hands
                    #     is_eye_hands_contact = time_seq_utils.gaze_to_adult_hands(
                    #         pred_phase_eyes_seq[-int(T_STATS_EYE_CONTACT * fps):], \
                    #         child_face_seq[-int(T_STATS_EYE_CONTACT * fps):],
                    #         adult_hands_seq[-int(T_STATS_EYE_CONTACT * fps) * 2:], T_STATS_EYE_CONTACT, fps)
                    #     # pred_phase_eyes_seq, abs_phase_eyes_seq = [], []
                    # if is_eye_contact:
                    #     cv2.putText(imgRGB, 'child-face-attention', (int(width / 2) - 60, int(height / 2) + 100),
                    #                 cv2.FONT_HERSHEY_PLAIN, font_size, (255, 192, 203), font_stroke, cv2.LINE_AA)
                    # if is_eye_hands_contact:
                    #     cv2.putText(imgRGB, 'child-hand-attention', (int(width / 2) - 60, int(height / 2) + 100),
                    #                 cv2.FONT_HERSHEY_PLAIN, font_size, (255, 192, 203), font_stroke, cv2.LINE_AA)

                        # cv2.putText(imgRGB, 'abs:'+str(abs_phase), (int(width/2)+40, int(height/2)-40), cv2.FONT_HERSHEY_PLAIN, font_size, (125, 255, 125), 5, cv2.LINE_AA)
                    # --------------- process mouth distance from face landmraks ------------- #
                    # child_lmks, adult_lmks = img_utils.match_face_lmks_to_person(face_lmks, adult_loc, child_loc,
                    #                                                              orig_img_rgb.shape)
                    # adult_mouth_dist, child_mouth_dist = img_utils.get_mouth_dist(adult_lmks, child_lmks,
                    #                                                               orig_img_rgb.shape)
                    # adult_mouth_seq.append(adult_mouth_dist), child_mouth_seq.append(child_mouth_dist)

                    if i % int(T_STATS_MOUTH * fps) == 0 and i > 0:  # statistics runs for every T_STATS_MOUTH sec
                        is_adult_speaks, is_child_speaks = False, False
                        is_adult_speaks, is_child_speaks = time_seq_utils.get_speaker_by_mouth(adult_mouth_seq,
                                                                                               child_mouth_seq)
                        adult_mouth_seq, child_mouth_seq = [], []

                    # if is_adult_speaks:
                    #   cv2.putText(imgRGB, 'adult speaks', (int(width/8), 30), cv2.FONT_HERSHEY_PLAIN, font_size, (255, 0, 0), font_stroke, cv2.LINE_AA)

                    # if is_child_speaks:
                    #   cv2.putText(imgRGB, 'child speaks', (int(width-width/5), 30), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 255), font_stroke, cv2.LINE_AA)

                    # ------------- run hands detector ------------- #
                    # hands_centers, fingers_tips = img_utils.get_hands_loc(orig_img_rgb, imgRGB)

                    # if (hands_centers is None) or (fingers_tips is None): # can't get hands keypoints
                    # Fallback to hands from skeleton

                    adult_left_hands = (pose_results[adult_rect_idx]['keypoints'][9][0], pose_results[adult_rect_idx]['keypoints'][9][1])
                    adult_right_hands = (pose_results[adult_rect_idx]['keypoints'][10][0],
                                        pose_results[adult_rect_idx]['keypoints'][10][1])
                    child_left_hands = (pose_results[child_rect_idx]['keypoints'][9][0], pose_results[child_rect_idx]['keypoints'][9][1])
                    child_right_hands = (pose_results[child_rect_idx]['keypoints'][10][0],
                                        pose_results[child_rect_idx]['keypoints'][10][1])

                    if pred_phase:
                        dy_face_left_hand = child_face_y - adult_left_hands[1]
                        dx_face_left_hand = child_face_x - adult_left_hands[0]
                        adult_hands_abs_phase_left = np.arctan2(dy_face_left_hand, dx_face_left_hand) * 180 / np.pi

                        dy_face_right_hand = child_face_y - adult_right_hands[1]
                        dx_face_right_hand = child_face_x - adult_right_hands[0]
                        adult_hands_abs_phase_right = np.arctan2(dy_face_right_hand, dx_face_right_hand) * 180 / np.pi
                        if adult_hands_abs_phase_left < 0 and adult_hands_abs_phase_right < 0:
                            adult_hands_abs_phase = np.max([adult_hands_abs_phase_left, adult_hands_abs_phase_right])
                        else:
                            adult_hands_abs_phase = np.min([adult_hands_abs_phase_left, adult_hands_abs_phase_right])
                        if adult_hands_abs_phase < 0:
                            abs_phase + 360

                    # left_hands = [(pose['keypoints'][9][0], pose['keypoints'][9][1]) for pose in pose_results if
                    #               pose['bbox'][-1] > 0.65]
                    # right_hands = [(pose['keypoints'][10][0], pose['keypoints'][10][1]) for pose in pose_results if
                    #                pose['bbox'][-1] > 0.65]

                    # if hands_centers:
                    # -------------------------- match hands to person ---------------------------- #
                    hands_dict = img_utils.arrange_hands_dict(adult_left_hands, adult_right_hands, child_left_hands, child_right_hands)
                    # hands_dict = img_utils.match_hands_to_person_polygon(masks, found_person_labels, adult_loc,
                    #                                                      child_loc, hands_centers)
                    # hands_dict, fingers_dict = img_utils.match_hands_to_person_diag(adult_loc, child_loc, hands_centers, fingers_tips)
                    adult_hands_loc = hands_dict['adult_hands'] if hands_dict['adult_hands'] else None
                    child_hands_loc = hands_dict['child_hands'] if hands_dict['child_hands'] else None
                    # break
                    # if adult_hands_loc is not None:
                    #     adult_hands_seq += adult_hands_loc
                    # if child_hands_loc is not None:
                    #     child_hands_seq += child_hands_loc

                    img_utils.draw_hands(imgRGB, child_hands_loc, adult_hands_loc, font_size, font_stroke)
                    # elif fingers_tips:
                    # hands_dict, fingers_dict = img_utils.match_fingers_to_person_polygon(masks, found_person_labels, adult_loc, child_loc, hands_centers, fingers_tips)
                    # ------------- collect fingers info; draw hands labels --------------- #
                    # adult_fingers, child_fingers, adult_hands, child_hands = img_utils.collect_and_draw_hands(imgRGB, fingers_dict, hands_dict,\
                    #                                                                         child_fingers_seq, adult_fingers_seq, font_size, font_stroke)
                    # adult_fingers_seq += adult_fingers
                    # child_fingers_seq += child_fingers

                    # ------------------- measure hands proximity and eye contact -------------- #
                    # if i % int(T_STATS_HANDS * fps) == 0:
                        # is_hands_interaction = False
                        # is_hands_interaction = HandInteraction.hands_proximity(child_fingers_seq, adult_fingers_seq)
                        # is_hands_interaction = HandInteraction.hands_proximity(child_hands_seq, adult_hands_seq)
                        # child_hands_seq, adult_hands_seq = [], []
                        # -------- # -------- #
                        # is_eye_contact = Interaction.gaze_estimation(child_fingers_seq, adult_fingers_seq)
                    # if is_hands_interaction:
                    #     # print('Found interaction')
                    #     cv2.putText(imgRGB, 'Hand Interaction!',
                    #                 (np.max([int((width / 2) - (width / 6)), 0]), int((height / 2) - (height / 6))),
                    #                 cv2.FONT_HERSHEY_PLAIN, font_size, (255, 255, 0), font_stroke, cv2.LINE_AA)
            if k >= int(T_STATS_ACTION_RECOGNITION * fps):    # i % int(T_STATS_ACTION_RECOGNITION * fps) == 0 and i > 0:  # statistics runs for every T_STATS_ACTION_RECOGNITION sec
                results = inference_recognizer(action_model, frames_dir)
                action = [(action_labels[k[0]], k[1]) for k in results]
                for f in glob.glob(frames_dir + '/*.jpg'):
                    os.remove(f)
                k = 1
            else:
                action = None
                img = cv2.cvtColor(orig_img_rgb, cv2.COLOR_RGB2BGR)
                if k > 9:
                    img_name = f'{frames_dir}\\img_000{k}.jpg'
                else:
                    img_name = f'{frames_dir}\\img_0000{k}.jpg'
                cv2.imwrite(img_name, img)
                k += 1

            if not pred_phase:
                pred_phase, abs_phase, center_of_child_face = None, None, None
            if not child_hands_loc:
                child_hands_loc, adult_hands_loc = None, None
            if not isinstance(child_loc,  (np.ndarray, np.generic)):
                child_loc, adult_loc = None, None
            out_df.loc[-1] = [timer, adult_loc, child_loc, center_of_child_face, adult_hands_loc, child_hands_loc, adult_mouth_dist,
                              child_mouth_dist, pred_phase, abs_phase, adult_hands_abs_phase, action, more_than_2_ppl]
            out_df.index = out_df.index + 1
            out_df = out_df.sort_index()  # sorting by index
            timer += time_bin
            img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
            output_video.write(img)
            i += 1
            print(i)
            #
            # if i > 250:
            #     break
        out_df.iloc[:] = out_df.iloc[::-1].values
        out_df.to_csv(out_name[:-3] + 'csv')
        output_video.release()
        cap.release()
        # ------ write both audio and video ----- #
        try:
            input_video = ffmpeg.input(out_name)
            input_audio = ffmpeg.input(video)
            ffmpeg.concat(input_video, input_audio, v=1, a=1).output(f'{out_name[:-4]}_audio.mp4').run(
                capture_stdout=True, capture_stderr=True, overwrite_output=True)
        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
            raise e





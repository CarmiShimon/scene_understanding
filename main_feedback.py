import os

import cv2
from AI_Feedback_utils import *
from Feedback_features import Features

if __name__ == '__main__':
    sub_dir_name = 'Child 103'  # 'play_pause' #  'Child 101'
    audio_dir = rf'../videos/orig_wavs_and_diarization/{sub_dir_name}'
    video_dir = rf'../videos/orig_videos/{sub_dir_name}'
    vid_csv_dir = rf'../videos/scene_understanding_videos/{sub_dir_name}'
    audio_csv_dir = rf'../videos/orig_wavs_and_diarization/{sub_dir_name}'
    out_dir = rf'../videos/modules/{sub_dir_name}'

    os.makedirs(out_dir, exist_ok=True)
    video_name = '3'
    is_audio_extract = False
    is_diarize = False  # Currently not supported
    is_merge_csvs = False  # merge csv of audio and video analysis
    is_find_features = False
    is_find_hizuk = True
    features = Features(win_len=0.5, overlap_prc=0.5, angle_delta=25, min_interaction_time=0.15)

    if is_audio_extract:
        extract_audio(video_dir, audio_dir)

    if is_diarize:
        diarize(audio_dir)

    if is_merge_csvs:
        # merge csv of audio and video analysis
        available_csv_files = csv_merger(vid_csv_dir, audio_csv_dir, out_dir)

    if is_find_features:
        extension = ['MOV', 'mp4', 'mkv', 'avi']
        for csv_file in glob.glob(f'{out_dir}/*AI.csv'):
            csv_file = csv_file.replace('\\', '/')
            print(f'processing {csv_file} for features')
            # if not csv_file.split('/')[-1] == f'{video_name}_AI.csv':
            #     continue
            # read video to get fps and width
            video_name = csv_file.split('/')[-1][:-7]
            for ext in extension:
                vid_filename = f'{video_dir}/{video_name}.{ext}'
                if os.path.isfile(vid_filename):
                    break

            cap = cv2.VideoCapture(vid_filename)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print('video fps = ', fps)
            if fps == 0.0:
                print(f'There is a problem with video {vid_filename}, check orig dir')
                break
            im_size = (int(cap.get(3)), int(cap.get(4)))  # width, height
            width, height = im_size[0], im_size[1]
            # overlaped_results_df = features.find_features(csv_file)
            # overlaped_results_df[::2].to_csv(f"{out_dir}/{csv_file.split('/')[-1][:-4]}_video_audio.csv")
            # results_df = smooth_overlaped_results(overlaped_results_df)
            results_df = features.find_features(csv_file, width, is_overlap=False, fps=fps)
            print(f"saving {out_dir}/{csv_file.split('/')[-1][:-4]}_video_audio.csv")
            results_df.to_csv(f"{out_dir}/{csv_file.split('/')[-1][:-4]}_video_audio.csv")


    if is_find_hizuk:
        resulted_csvs = glob.glob(f'{out_dir}/*AI_video_audio.csv')
        for csv_file in resulted_csvs:
            csv_file = csv_file.replace('\\', '/')
            print(f'processing {csv_file} for hizuk')
            # video_name = csv_file.split('/')[-1][:-7]
            # if not csv_file.split('/')[-1] == f'{video_name}_AI_video_audio.csv':
            #     continue
            df = pd.read_csv(csv_file)
            reinforcement, communicative_speech = features.find_hizuk(df, time_res=df['start_time'].values[1])
            reinforcement = features.smooth_reinforcement(reinforcement)
            print(reinforcement)
            df['reinforcement'] = reinforcement
            df['communicative_speech'] = communicative_speech
            df.drop('Unnamed: 0', inplace=True, axis=1)
            df.to_csv(csv_file)

        # later:
        # find objects in hands
        # train NN for Diarization
        # add gaze to adult hand phase to csv
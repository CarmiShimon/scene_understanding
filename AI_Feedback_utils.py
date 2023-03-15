import moviepy.editor as mp
import glob
import os
import pandas as pd
import numpy as np
import itertools

def extract_audio(video_dir, audio_dir):
    video_types = [f'{video_dir}/*.MOV', f'{video_dir}/*.mp4', f'{video_dir}/*.mkv', f'{video_dir}/*.avi']
    videos_grabbed = list(itertools.chain(*[glob.glob(v) for v in video_types]))
    os.makedirs(audio_dir, exist_ok=True)
    for vid in videos_grabbed:
        vid = vid.replace('\\', '/')
        clip = mp.VideoFileClip(vid)
        clip.audio.write_audiofile(f"{audio_dir}/{vid.split('/')[-1][:-4]}.wav")

def diarize(audio_dir):
    for wav in glob.glob(audio_dir + '/*.wav'):
        print('wav name: ', wav)
        vad = pipeline(wav)

    embeddings = []
    for turn, _, speaker in vad.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        excerpt = Segment(turn.start, turn.end)
        embedding = inference.crop(wav, excerpt)
        embeddings.append(embedding)

    # TODO: train classifier on embedding of kids and adults
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)
    # result = 1 - spatial.distance.cosine(embeddings[0], embeddings[3])
    # TODO: save csv file of diarization

def csv_merger(vid_csv_dir, audio_csv_dir, out_dir):
    available_csv_files = []

    for vid_csv in glob.glob(vid_csv_dir + '/*.csv'):
        vid_csv = vid_csv.replace('\\', '/')
        diarization_filepath = f"{audio_csv_dir}/{vid_csv.split('/')[-1][:-7]}_child_adult_speech.csv"
        if not os.path.isfile(diarization_filepath):
            print(f'{diarization_filepath} does not exist')
            continue
        print(f'Processing {diarization_filepath}')
        vid_df = pd.read_csv(vid_csv)
        audio_df = pd.read_csv(diarization_filepath)
        speaker = [None] * len(vid_df)

        audio_df['start_usec'] = audio_df['start_usec']/1e6
        audio_df['end_usec'] = audio_df['end_usec']/1e6
        speaker = [None] * len(vid_df)
        time_bin = vid_df.iloc[1]['time']
        for idx, row in audio_df.iterrows():
            start_idx = int(np.round(row['start_usec']/time_bin))
            end_idx = int(np.round(row['end_usec']/time_bin))
            spk = row['speaker_estimate']
            try:
                for idx in range(start_idx, end_idx):
                    speaker[idx] = spk
            except:
                print(f'An error has occured in {vid_csv}')
        vid_df['speaker'] = speaker
        vid_df.to_csv(f"{out_dir}/{vid_csv.split('/')[-1]}")
        available_csv_files.append(vid_csv)
    return available_csv_files


def smooth_overlaped_results(df):
    for i in range(len(df)):
        spk_counts = df.iloc[i:1+2]['speaker'].value_counts()
        print(spk_counts)
        face_counts = df.iloc[i:1+2]['face_looking'].value_counts()
        print(face_counts)
        hand_counts = df.iloc[i:1+2]['hand_looking'].value_counts()
        print(face_counts)
        hands_counts = df.iloc[i:1+2]['hand_interaction'].value_counts()
        print(hands_counts)
        break

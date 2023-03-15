import os

from moviepy.editor import *
import datetime
import itertools
import glob
import pandas as pd
import numpy as np
import csv

class Summarize:
    def __init__(self, size, vid_duration, child_adult_seconds, duration=3):
        self.summary_dict = {'Reinforcement': 0, 'No-Reinforcement': 0, 'Double Demand': 0, 'Communicative Speech': 0}
        self.size = size
        self.duration = duration
        self.vid_duration = vid_duration
        self.child_adult_seconds = child_adult_seconds

    def color_clip(self, start, end, results):
        # creating a color clip
        clip = ColorClip(size=self.size, color=[255, 255, 255], duration=self.duration)
        # Generate a text clip
        # setting position of text in the center and duration will be 3 seconds
        if results['reinforcement'] == 1:
            txt_clip = TextClip("Reinforcement", fontsize=75, color='black')
            self.summary_dict['Reinforcement'] += 1
            self.summary_dict['Communicative Speech'] += 1
        elif results['reinforcement'] == 2:
            txt_clip = TextClip("No-Reinforcement", fontsize=75, color='black')
            self.summary_dict['No-Reinforcement'] += 1
            self.summary_dict['Communicative Speech'] += 1
        elif results['reinforcement'] == 3:
            txt_clip = TextClip("No-Reinforcement - Double Demand", fontsize=75, color='black')
            self.summary_dict['Double Demand'] += 1
            self.summary_dict['Communicative Speech'] += 1
        elif results['communicative_speech'] == 1:
            txt_clip = TextClip("Communicative Speech", fontsize=75, color='black')
            self.summary_dict['Communicative Speech'] += 1

        txt_clip = txt_clip.set_pos('center').set_duration(self.duration)

        # set times #
        txt_clip2 = TextClip(
            f'{str(datetime.timedelta(seconds=int(start)))} : {str(datetime.timedelta(seconds=int(end)))}', fontsize=75,
            color='blue')
        txt_clip2 = txt_clip2.set_pos('bottom').set_duration(self.duration)
        # Overlay the text clip on the first video clip
        video = CompositeVideoClip([clip, txt_clip, txt_clip2], size=self.size)
        return video

    def clip_summary(self):
        clip = ColorClip(size=self.size, color=[255, 255, 255], duration=10)
        r_times = self.summary_dict['Reinforcement']
        r_stats = round(r_times / (self.child_adult_seconds / 60), 1)
        nr_times = self.summary_dict['No-Reinforcement']
        dd_times = self.summary_dict['Double Demand']
        nr_stats = round(
            (self.summary_dict['No-Reinforcement'] + self.summary_dict['Double Demand']) / (self.child_adult_seconds / 60), 1)
        cc = self.summary_dict['Communicative Speech']
        cc_stats = round(cc / (self.child_adult_seconds / 60), 1)
        txt = f"Reinforcement {str(r_times)} times, ({r_stats} per valid frames)"
        txt_clip = TextClip(txt, fontsize=55, color='black', method='caption')
        txt_clip = txt_clip.set_pos(('center', 'top')).set_duration(10)
        txt2 = f"No Reinforcement {str(nr_times)} times, Double Demand {str(dd_times)} times, ({nr_stats} per valid frames)"
        txt_clip2 = TextClip(txt2, fontsize=55, color='black', method='caption')
        txt_clip2 = txt_clip2.set_pos('center').set_duration(10)
        if self.summary_dict['Communicative Speech'] > 0:
            txt3 = f"Child communicated with you {str(cc)} times, ({cc_stats} per valid frames)"
        else:
            txt3 = f"Your child still haven't communicated with you"
        txt_clip3 = TextClip(txt3, fontsize=55, color='blue', method='caption')
        txt_clip3 = txt_clip3.set_pos(("center", 'bottom')).set_duration(10)
        video = CompositeVideoClip([clip, txt_clip, txt_clip2, txt_clip3], size=self.size)
        return video


if __name__ == '__main__':
    sub_dir_name = 'Child 103'
    video_dir = f'../videos/orig_videos/{sub_dir_name}'
    audio_dir = f'../videos/orig_wavs_and_diarization{sub_dir_name}'
    modules_dir = f'../videos/modules/{sub_dir_name}'
    video_types = [f'{video_dir}/*.MOV', f'{video_dir}/*.mov', f'{video_dir}/*.mp4', f'{video_dir}/*.avi']

    out_dir = f'../videos/summary/{sub_dir_name}'
    os.makedirs(out_dir, exist_ok=True)
    # ------------------------------------------------------------- #
    videos_grabbed = list(itertools.chain(*[glob.glob(v) for v in video_types]))
    for video in videos_grabbed:
        print('processing video: ', video)
        video = video.replace('\\', '/')
        vid_name = video.split('/')[-1]
        print(vid_name)
        # if not (vid_name == '5A.mov'):# or vid_name == '20.mp4'):
        #     continue
        clip = VideoFileClip(video)
        fps = clip.fps
        video_duration = clip.duration
        video_size = clip.size
        # ---------- load results df file ---------- #

        results_csv = modules_dir + '/' + video.split('/')[-1].split('.')[0] + '_AI_video_audio.csv'
        results_df = pd.read_csv(results_csv)
        summarize = Summarize(video_size, video_duration, child_adult_seconds=results_df['2_ppl_seconds'].iloc[0], duration=3)
        # Replace the filename below.
        required_video_file = video

        indices = results_df.index[
            (results_df['communicative_speech'] != 0) | (results_df['reinforcement'] != 0)].tolist()
        # print(indices)
        res_indices = indices
        times = results_df.iloc[indices]['start_time'].values
        if len(times) < 5:
            print('There are no interactions')
            continue
        start_time = times[0]
        i = 1
        sub_clips = []
        while i < len(times):
            if np.diff(times).max() < 2:  # in case there is only 1 interesting interaction
                end_time = times[-1]
                white_clip = summarize.color_clip(start_time, end_time, results_df.iloc[indices[i - 1]])
                sub_clips.append(white_clip)
                sub_clips.append(clip.subclip(np.max([start_time - 1, 0]), np.min([end_time + 1, video_duration])).crossfadein(1))
                break
            elif times[i] > times[i - 1] + 3:  # in case there are more than 1 interesting interaction
                end_time = times[i - 1]
                white_clip = summarize.color_clip(start_time, end_time, results_df.iloc[indices[i - 1]])
                sub_clips.append(white_clip)
                sub_clips.append(
                    clip.subclip(np.max([start_time - 1, 0]), np.min([end_time + 1, video_duration])).crossfadein(1))
                start_time = times[i]
                i += 1
            else:
                i += 1
                continue

        # write to csv file
        out_dict = summarize.summary_dict
        out_dict['child_adult_seconds'] = summarize.child_adult_seconds
        df = pd.DataFrame.from_dict(out_dict, orient='index')
        df.to_csv(f"{out_dir}/{video.split('/')[-1].split('.')[0]}.csv")

        sub_clips.append(summarize.clip_summary())
        final = concatenate_videoclips(sub_clips)
        final.write_videofile(f"{out_dir}/{video.split('/')[-1].split('.')[0]}.mp4", fps=fps)



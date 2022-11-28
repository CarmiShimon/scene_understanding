import os
import itertools
import ass
import glob
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import pandas as pd

def convert_videos(input_dir):
    video_types = [f'{input_dir}\*.MOV', f'{input_dir}\*.mov', f'{input_dir}\*.mkv', f'{input_dir}\*.avi']
    videos_grabbed = list(itertools.chain(*[glob.glob(v) for v in video_types]))
    for vid_file in videos_grabbed:
        try:
            os.system(f'ffmpeg -i {vid_file} {vid_file[:-4]}.mp4')
            # os.remove(vid_file)
        except: f'there was an error with {vid_file}'


def split_video(main_dir, sub_dict, out_dir):
    for i, (filename, starttime, endtime, label) in enumerate(zip(sub_dict['vid_name'], sub_dict['start_time'], sub_dict['end_time'], sub_dict['label'])):
        required_video_file = main_dir + '\\' + filename + '.mp4'
        target_filename = f'{out_dir}\\{filename}_{i}_{label}.mp4'
        ffmpeg_extract_subclip(required_video_file, starttime, endtime, targetname=target_filename)

def read_subtitles(subtitles_dir):
    d = {'vid_name': [], 'start_time': [], 'end_time': [], 'label': []}
    for i, sub_file in enumerate(glob.glob(subtitles_dir + '/*.ass')):
        vid_name = sub_file.split('\\')[-1].split('.ass')[0]
        with open(sub_file, encoding='utf_8_sig') as f:
            doc = ass.parse(f)
            for event in doc.events:
                if event.text:
                    d['vid_name'].append(vid_name)
                    d['label'].append(event.text)
                    d['start_time'].append(event.start.seconds + event.start.microseconds/1000000)
                    d['end_time'].append(event.end.seconds + event.end.microseconds/1000000)
    return d



if __name__ == '__main__':
    out_dir = r'C:\Users\carmi\PycharmProjects\socailmind\scene_understanding\videos\orig_videos\splitted_videos'
    main_dir = r'C:\Users\carmi\PycharmProjects\socailmind\scene_understanding\videos\orig_videos\Child 103'
    convert_videos(main_dir)
    # subtitle_dict = read_subtitles(main_dir)
    # split_video(main_dir, subtitle_dict, out_dir)
import os.path
import os
import cv2
import itertools
import glob
import pandas as pd
import numpy as np
import ffmpeg


def get_video_params(vid_filename):
    cap = cv2.VideoCapture(vid_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_bin = 1/fps
    im_size = (int(cap.get(3)), int(cap.get(4))) # width, height
    font_size = np.max([im_size[0]/600, 1.2])
    font_stroke = int(min(im_size[0], im_size[1]) / 130)
    return cap, fps, time_bin, im_size, font_size, font_stroke


if __name__ == '__main__':
    sub_dir_name = 'Child 103'  # 'play_pause'  # 'Child 101'
    video_dir = f'../videos/orig_videos/{sub_dir_name}'
    audio_dir = f'../videos/orig_wavs_and_diarization/{sub_dir_name}'
    modules_dir = f'../videos/modules/{sub_dir_name}'
    video_types = [f'{video_dir}/*.MOV', f'{video_dir}/*.mp4', f'{video_dir}/*.avi', f'{video_dir}/*.mov']
    # ------------------------------------------------------------- #
    videos_grabbed = list(itertools.chain(*[glob.glob(v) for v in video_types]))
    for video in videos_grabbed:
        video = video.replace('\\', '/')
        print('processing video: ', video)
        vid_name = video.split('/')[-1]
        print(vid_name)
        if not (vid_name == '2.mp4'):# or vid_name == '6a.mp4'):
            continue
        # if not (vid_name == '3.mov'):

        # if not (vid_name == '9b.mp4'):
        #     continue
        # ---------- load results df file ---------- #
        results_csv = f"{modules_dir}/{vid_name[:-4]}_AI_video_audio.csv"
        if not os.path.isfile(results_csv):
            continue
        results_df = pd.read_csv(results_csv)

        # -------------------------------------- #
        # loading video file clip
        cap, fps, time_bin, im_size, font_size, font_stroke = get_video_params(video)
        width, height = im_size[0], im_size[1]
        out_name = f'{results_csv[:-4]}.mp4'
        output_video = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'MP4V'), fps, im_size)
        # break

        total_rows = len(results_df)
        t_frame = 0
        i_rows = 0
        i = 0
        boldness = 3
        # ---------iterate over frames--------- #
        while (cap.isOpened()):
            success, img = cap.read()
            if not success:
                print("No more frames to process.")
                # If loading a video, use 'break' instead of 'continue'.
                break
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if video[-3:] == 'MOV' or video[-3:] == 'mov':
                imgRGB = cv2.flip(imgRGB, 0)
                imgRGB = cv2.flip(imgRGB, 1)
            # if video == '../videos/orig_videos/play_pause/6.mp4' or video == '../videos/orig_videos/play_pause/8.mp4':
            #     imgRGB = cv2.flip(imgRGB, 0)
            #     imgRGB = cv2.flip(imgRGB, 1)

            if i_rows > 1:
                time_res = results_df.iloc[i_rows]['start_time'] - results_df.iloc[i_rows - 1]['start_time']
            else:
                time_res = results_df.iloc[1]['start_time']

            if (t_frame >= results_df.iloc[i_rows]['start_time']) and (
                    t_frame <= results_df.iloc[i_rows]['start_time'] + time_res):
                # check first for 3 ppl and more
                if False:  # results_df.iloc[i_rows]['three_ppl']:
                    cv2.putText(imgRGB, 'Warning: More than 2 people in scene',
                                (int(width / 2) - int(width / 20), int(height / 10)), cv2.FONT_HERSHEY_PLAIN,
                                font_size * 2, (0, 0, 0), 2, cv2.LINE_AA)
                else:
                    if not str(results_df.iloc[i_rows]['speaker']) == 'nan':
                        # speaker_sent = f"Speaker: {results_df.iloc[i_rows]['speaker']}"
                        # Dana asked ignoring adult speaks label
                        # if str(results_df.iloc[i_rows]['speaker']) == 'a':
                        #   cv2.putText(imgRGB, speaker_sent, (int(width/2)-int(width/10), int(height/10)), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 255), 2, cv2.LINE_AA)
                        if str(results_df.iloc[i_rows]['speaker']) == 'b' or str(results_df.iloc[i_rows]['speaker']) == 'c':
                            # speaker_sent = f"Speaker: {results_df.iloc[i_rows]['speaker']}"
                            speaker_sent = 'Child Speaks!'
                            cv2.putText(imgRGB, speaker_sent, (int(width / 2) - int(width / 20), int(height / 10)),
                                        cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 0), boldness + 3, cv2.LINE_AA)
                            cv2.putText(imgRGB, speaker_sent, (int(width / 2) - int(width / 20), int(height / 10)),
                                        cv2.FONT_HERSHEY_PLAIN, font_size, (255, 234, 0), boldness, cv2.LINE_AA)
                    # if results_df.iloc[i_rows]['hand_looking'] == 1:
                    #     if (i_rows < total_rows - 2) and results_df.iloc[i_rows - 1]['hand_looking'] == 0 and \
                    #             results_df.iloc[i_rows + 1]['hand_looking'] == 0:
                    #         results_df.at[
                    #             i_rows + 1, 'hand_looking'] = 1  # force next value for extended time visualization
                    #     hand_looking_sent = f'Child is looking at Adult Hands'
                    #     cv2.putText(imgRGB, hand_looking_sent, (int(width / 2) - int(width / 10), int(height / 6)),
                    #                 cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 0), boldness + 3, cv2.LINE_AA)
                    #     cv2.putText(imgRGB, hand_looking_sent, (int(width / 2) - int(width / 10), int(height / 6)),
                    #                 cv2.FONT_HERSHEY_PLAIN, font_size, (255, 234, 0), boldness, cv2.LINE_AA)
                    # if results_df.iloc[i_rows]['face_looking'] == 1:
                    #     if (i_rows < total_rows - 2) and results_df.iloc[i_rows - 1]['face_looking'] == 0 and \
                    #             results_df.iloc[i_rows + 1]['face_looking'] == 0:
                    #         results_df.at[
                    #             i_rows + 1, 'face_looking'] = 1  # force next value for extended time visualization
                    #     face_looking_sent = f'Child is looking at Adult Face'
                    #     cv2.putText(imgRGB, face_looking_sent, (int(width / 2) - int(width / 10), int(height / 4)),
                    #                 cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 0), boldness + 3, cv2.LINE_AA)
                    #     cv2.putText(imgRGB, face_looking_sent, (int(width / 2) - int(width / 10), int(height / 4)),
                    #                 cv2.FONT_HERSHEY_PLAIN, font_size, (255, 234, 0), boldness, cv2.LINE_AA)
                    # if results_df.iloc[i_rows]['hand_interaction'] == 1:
                    #     if (i_rows < total_rows - 2) and results_df.iloc[i_rows - 1]['hand_interaction'] == 0 and \
                    #             results_df.iloc[i_rows + 1]['hand_interaction'] == 0:
                    #         results_df.at[
                    #             i_rows + 1, 'hand_interaction'] = 1  # force next value for extended time visualization
                    #     hands_sent = f'Hands interaction'
                    #     cv2.putText(imgRGB, hands_sent, (int(width / 2) - int(width / 10), int(height / 2)),
                    #                 cv2.FONT_HERSHEY_PLAIN, font_size, (0, 0, 0), boldness + 3, cv2.LINE_AA)
                    #     cv2.putText(imgRGB, hands_sent, (int(width / 2) - int(width / 10), int(height / 2)),
                    #                 cv2.FONT_HERSHEY_PLAIN, font_size, (255, 234, 0), boldness, cv2.LINE_AA)
                    if results_df.iloc[i_rows]['reinforcement'] == 1:
                        reinf_sent = f'REINFORCEMENT'
                        cv2.putText(imgRGB, reinf_sent,
                                    (int(width / 2) - int(width / 10), int(height) - int(height / 10)),
                                    cv2.FONT_HERSHEY_PLAIN, font_size * 1.5, (255, 255, 255), boldness + 3, cv2.LINE_AA)
                        cv2.putText(imgRGB, reinf_sent,
                                    (int(width / 2) - int(width / 10), int(height) - int(height / 10)),
                                    cv2.FONT_HERSHEY_PLAIN, font_size * 1.5, (0, 255, 0), boldness, cv2.LINE_AA)
                    if results_df.iloc[i_rows]['reinforcement'] == 2:
                        reinf_sent = f'NO-REINFORCEMENT'
                        cv2.putText(imgRGB, reinf_sent,
                                    (int(width / 2) - int(width / 10), int(height) - int(height / 10)),
                                    cv2.FONT_HERSHEY_PLAIN, font_size * 1.5, (255, 255, 255), boldness + 3, cv2.LINE_AA)
                        cv2.putText(imgRGB, reinf_sent,
                                    (int(width / 2) - int(width / 10), int(height) - int(height / 10)),
                                    cv2.FONT_HERSHEY_PLAIN, font_size * 1.5, (255, 0, 0), boldness, cv2.LINE_AA)
                    if results_df.iloc[i_rows]['reinforcement'] == 3:
                        reinf_sent = f'NO-REINFORCEMENT'
                        # Draw black background rectangle
                        cv2.putText(imgRGB, reinf_sent,
                                    (int(width / 2) - int(width / 10), int(height) - int(height / 10)),
                                    cv2.FONT_HERSHEY_PLAIN, font_size * 1.5, (255, 255, 255), boldness + 3, cv2.LINE_AA)
                        cv2.putText(imgRGB, reinf_sent,
                                    (int(width / 2) - int(width / 10), int(height) - int(height / 10)),
                                    cv2.FONT_HERSHEY_PLAIN, font_size * 1.5, (255, 0, 0), boldness, cv2.LINE_AA)
                        cv2.putText(imgRGB, 'DOUBLE DEMAND',
                                    (int(width / 2) - int(width / 10), int(height) - int(height / 22)),
                                    cv2.FONT_HERSHEY_PLAIN, font_size * 1.5, (255, 255, 255), boldness + 3, cv2.LINE_AA)
                        cv2.putText(imgRGB, 'DOUBLE DEMAND',
                                    (int(width / 2) - int(width / 10), int(height) - int(height / 22)),
                                    cv2.FONT_HERSHEY_PLAIN, font_size * 1.5, (255, 0, 0), boldness, cv2.LINE_AA)
                    if results_df.iloc[i_rows]['communicative_speech'] == 1:
                        comm_sent = f'COMMUNICATIVE SPEECH'
                        cv2.putText(imgRGB, comm_sent,
                                    (int(width / 2) - int(width / 10), int(height) - int(height / 6)),
                                    cv2.FONT_HERSHEY_PLAIN, font_size * 1.5, (255, 255, 255), boldness + 3, cv2.LINE_AA)
                        cv2.putText(imgRGB, comm_sent,
                                    (int(width / 2) - int(width / 10), int(height) - int(height / 6)),
                                    cv2.FONT_HERSHEY_PLAIN, font_size * 1.5, (0, 255, 0), boldness, cv2.LINE_AA)
                    # if isinstance(results_df.iloc[i_rows]['action'], str):
                    #     cv2.putText(imgRGB, results_df.iloc[i_rows]['action'],
                    #                 (int(width / 2) - int(width / 10), int(height) - int(height / 6)),
                    #                 cv2.FONT_HERSHEY_PLAIN, font_size * 1.5, (0, 0, 255), boldness, cv2.LINE_AA)
            else:
                if i_rows < total_rows - 1:
                    i_rows += 1

            i += 1
            # ---- delete later ---- #
            # if t_frame > 15:
            #   break

            t_frame += time_bin
            img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
            output_video.write(img)
            if i % 100 == 0:
                print(i)
        output_video.release()
        cap.release()

        # ------ write both audio and video ----- #
        try:
            input_video = ffmpeg.input(out_name)
            input_audio = ffmpeg.input(f"{audio_dir}/{video.split('/')[-1][:-4]}.wav")
            ffmpeg.concat(input_video, input_audio, v=1, a=1).output(f'{out_name[:-4]}_with_audio.mp4').run(
                capture_stdout=True, capture_stderr=True, overwrite_output=True)
        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
            raise e


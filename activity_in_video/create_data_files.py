import os
import glob


def create_train_test_val_files(main_dir, out_files, split_prc=0.1):
    train_lines = []
    val_lines = []
    test_lines = []
    for i, sub_dir in enumerate(os.listdir(main_dir)):
        files = os.listdir(os.path.join(main_dir, sub_dir))
        total_files = len(files)
        for k, file in enumerate(files):
            if k < int((1 - 2 * split_prc) * total_files):
                train_lines.append(f'{sub_dir}\\{file} {i + 1}\n')
            elif k >= int((1 - 2 * split_prc) * total_files) and k < int((1 - split_prc) * total_files):
                val_lines.append(f'{sub_dir}\\{file} {i+1}\n')
            else:
                test_lines.append(f'{sub_dir}\\{file} {i+1}\n')


    with open(out_files[0], 'w') as f:
        f.writelines(train_lines)
    with open(out_files[1], 'w') as f:
        f.writelines(val_lines)
    with open(out_files[2], 'w') as f:
        f.writelines(test_lines)

if __name__ == '__main__':
    # main_dir = r'C:\Users\carmi\PycharmProjects\socailmind\scene_understanding\videos\orig_videos\play_pause'
    main_dir = r'C:\Users\carmi\PycharmProjects\activity_carmi\data\kinetics400'
    out_files = [r'C:\Users\carmi\PycharmProjects\activity_carmi\data\kinetics400_train_list_videos.txt',
                r'C:\Users\carmi\PycharmProjects\activity_carmi\data\kinetics400_val_list_videos.txt',
                r'C:\Users\carmi\PycharmProjects\activity_carmi\data\kinetics400_test_list_videos.txt']
    create_train_test_val_files(main_dir, out_files, split_prc=0.1)
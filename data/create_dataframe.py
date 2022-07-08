import os
from tqdm import tqdm
import pandas as pd

id2label = {
    '0': 'xoay ghế trái',
    '1': 'xoay ghế phải',
    '2': 'bật đèn',
    '3': 'bật đèn lên',
    '4': 'tắt đèn',
    '5': 'tắt đèn đi',
    '6': 'sáng quá',
    '7': 'tối quá',
    '8': 'bật nhạc',
    '9': 'bật nhạc lên',
    '10': 'dừng nhạc',
    '11': 'chuyển nhạc',
    '12': 'bật màn hình',
    '13': 'tắt màn hình',
    '14': 'bật laptop',
    '15': 'tắt laptop',
    '16': 'bật tv',
    '17': 'tắt tv',
}

all_commands = [key[0] for key in id2label.items()]


def create_dataframe(all_data_dir):
    data_dict = {'filepath': [], 'transcription': []}

    for dir in tqdm(os.listdir(all_data_dir)):
        sub_folder_dir = os.path.join(all_data_dir, dir)
        for command in all_commands:
            wav_dir = os.path.join(sub_folder_dir, command)
            wav_paths = [os.path.join(wav_dir, filepath) for filepath in os.listdir(wav_dir) if
                         filepath.endswith('.wav')]
            data_dict['filepath'].extend(wav_paths)
            data_dict['transcription'].extend([id2label[command]] * len(wav_paths))

    df = pd.DataFrame.from_dict(data=data_dict)
    df.sort_values(by='transcription', inplace=True, ignore_index=True)

    return df


if __name__ == "__main__":
    df = create_dataframe('/home/hoang/PycharmProjects/Speech_Recognition/all_data')
    print(df.iloc[1500]['filepath'])


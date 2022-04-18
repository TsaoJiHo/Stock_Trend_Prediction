from tqdm import tqdm
import pandas as pd
import pathlib

def oversampling(path, label):
    label_loc_map = {'15d': 25, '30d': 26, '90d': 27, '180d': 28}
    df = pd.read_csv(path)
    df = df.iloc[:, list(range(25)) + [label_loc_map[label]]]
    df = df.dropna()
    
    df_0 = df[df[label] == 0]
    df_1 = df[df[label] == 1]
    df_2 = df[df[label] == 2]
    split_gap_0 = int(len(df_0) * 0.8)
    split_gap_1 = int(len(df_1) * 0.8)
    split_gap_2 = int(len(df_2) * 0.8)
    train_df = pd.concat([df_0[:split_gap_0], df_1[:split_gap_1], df_2[:split_gap_2]])
    test_df = pd.concat([df_0[split_gap_0:], df_1[split_gap_1:], df_2[split_gap_2:]])

    for i, df in enumerate([train_df, test_df]):
        dataset = {0: 'training_data', 1: 'testing_data'}
        value_counts_list = [df[label].value_counts()[0], df[label].value_counts()[1], df[label].value_counts()[2]]

        down_dif = max(value_counts_list) - value_counts_list[0]
        stable_dif = max(value_counts_list) - value_counts_list[1]
        up_dif = max(value_counts_list) - value_counts_list[2]


        resample_df = [df, df[df[label] == 0].sample(n = down_dif, replace=(value_counts_list[0] < down_dif))
        , df[df[label] == 1].sample(n = stable_dif, replace=(value_counts_list[1] < stable_dif))
        , df[df[label] == 2].sample(n = up_dif, replace=(value_counts_list[2] < up_dif))]


        pd.concat(resample_df).to_csv(pathlib.Path(path.parent/(f'{dataset[i]}_{path.stem.split("_")[-1]}_{label}')).with_suffix('.csv'))
        # print(path.stem)


if __name__ == '__main__':

    for path in tqdm(pathlib.Path('./data/training_data').glob('*.csv')):
        for label in ['15d', '30d', '90d', '180d']:
            try:
                oversampling(path, label)
            except Exception as e:
                print(path.stem, label, str(e))

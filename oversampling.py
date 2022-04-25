from tqdm import tqdm
import pandas as pd
import pathlib

def oversampling(path, label):
    label_loc_map = {'10d': 25, '20d': 26, '60d': 27, '120d': 28}
    df = pd.read_csv(path)
    df = df.iloc[:, list(range(25)) + [label_loc_map[label]]]
    df = df.dropna()
    
    df_0 = df[df[label] == 0]
    df_1 = df[df[label] == 1]
    df_2 = df[df[label] == 2]
    split_train_gap_0 = int(len(df_0) * 0.7)
    split_train_gap_1 = int(len(df_1) * 0.7)
    split_train_gap_2 = int(len(df_2) * 0.7)
    split_val_gap_0 = int(len(df_0) * 0.85)
    split_val_gap_1 = int(len(df_1) * 0.85)
    split_val_gap_2 = int(len(df_2) * 0.85)
    train_df = pd.concat([df_0[:split_train_gap_0], df_1[:split_train_gap_1], df_2[:split_train_gap_2]])
    val_df = pd.concat([df_0[split_train_gap_0:split_val_gap_0], 
                        df_1[split_train_gap_1:split_val_gap_1], 
                        df_2[split_train_gap_2:split_val_gap_2]])
    test_df = pd.concat([df_0[split_val_gap_0:], df_1[split_val_gap_1:], df_2[split_val_gap_2:]])

    for i, df in enumerate([train_df, val_df, test_df]):
        dataset = {0: 'training_data', 1:'val_data', 2: 'testing_data'}
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
        for label in ['10d', '20d', '60d', '120d']:
            try:
                oversampling(path, label)
            except Exception as e:
                print(path.stem, label, str(e))

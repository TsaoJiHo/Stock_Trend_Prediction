from tqdm import tqdm
import pandas as pd
import pathlib

def oversampling(path, label):
    label_loc_map = {'15d': 25, '30d': 26, '90d': 27, '180d': 28}
    df = pd.read_csv(path)
    df = df.iloc[:, list(range(25)) + [label_loc_map[label]]]
    d = int(label[:-1])
    if 'testing' in path.stem:
        df = df[:-int(d * 0.2)]
    elif 'training' in path.stem:
        df = df[:-int(d * 0.8)]

    value_counts_list = [df[label].value_counts()[0], df[label].value_counts()[1], df[label].value_counts()[2]]

    down_dif = max(value_counts_list) - value_counts_list[0]
    stable_dif = max(value_counts_list) - value_counts_list[1]
    up_dif = max(value_counts_list) - value_counts_list[2]

    if value_counts_list[0] < down_dif:
        resample_df = [df, df[df[label] == 0].sample(n = down_dif, replace=True)
        , df[df[label] == 1].sample(n = stable_dif)
        , df[df[label] == 2].sample(n = up_dif)]
    else:
        resample_df = [df, df[df[label] == 0].sample(n = down_dif)
        , df[df[label] == 1].sample(n = stable_dif)
        , df[df[label] == 2].sample(n = up_dif)]

    pd.concat(resample_df).to_csv(pathlib.Path(path.parent/(path.stem + f'_{label}')).with_suffix('.csv'))
    # print(path.stem)


if __name__ == '__main__':

    for path in tqdm(pathlib.Path('./data/training_data').glob('*.csv')):
        for label in ['15d', '30d', '90d', '180d']:
            oversampling(path, label)
    
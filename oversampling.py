import pandas as pd

def oversampling(path, label, label_loc):
    df = pd.read_csv(path)
    df = df.iloc[:, [0,1,2,3,4,5,6,7,8,9,label_loc]]

    value_counts_list = [df[label].value_counts()[0], df[label].value_counts()[1], df[label].value_counts()[2]]

    down_dif = max(value_counts_list) - value_counts_list[0]
    stable_dif = max(value_counts_list) - value_counts_list[1]
    up_dif = max(value_counts_list) - value_counts_list[2]


    resample_df = [df, df[df[label] == 0].sample(n = down_dif)
    , df[df[label] == 1].sample(n = stable_dif)
    , df[df[label] == 2].sample(n = up_dif)]

    pd.concat(resample_df).to_csv(path[:-4] + '_' + label + '.csv')

if __name__ == '__main__':
    oversampling('./testing_data_v2.csv', '30d', 14)
    oversampling('./testing_data_v2.csv', '15d', 13)
    oversampling('./testing_data_v2.csv', '7d', 12)
    oversampling('./testing_data_v2.csv', '3d', 11)
    oversampling('./testing_data_v2.csv', '1d', 10)

    oversampling('./training_data_v2.csv', '30d', 14)
    oversampling('./training_data_v2.csv', '15d', 13)
    oversampling('./training_data_v2.csv', '7d', 12)
    oversampling('./training_data_v2.csv', '3d', 11)
    oversampling('./training_data_v2.csv', '1d', 10)
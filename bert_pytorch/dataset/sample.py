from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


def generate_pairs(line, window_size):
    line = np.array(line)
    line = line[:, 0]

    seqs = []
    for i in range(0, len(line), window_size):
        seq = line[i:i + window_size]
        seqs.append(seq)
    seqs += []
    seq_pairs = []
    for i in range(1, len(seqs)):
        seq_pairs.append([seqs[i - 1], seqs[i]])
    return seqs


def fixed_window(line, window_size, adaptive_window, seq_len=None, min_len=0, is_label=False, last_seq_id = 0):
    '''
    process one line into multiple sequences
    :param line:
    :type line:
    :param window_size:
    :type window_size:
    :param adaptive_window:
    :type adaptive_window:
    :param seq_len:
    :type seq_len:
    :param min_len:
    :type min_len:
    :param is_label:
    :type is_label:
    :return:
    :rtype:
    '''
    seq_label = 0
    token_labels = []
    if is_label:
        records = line.split(",")
        attr_size = len(records)
        seq_label = int(records[1])
        token_labels = [int(v) for v in records[2].split()]
        line =  [ln.split(",") for ln in records[3].split()]#split()以空格为分隔符，包括\n

        if attr_size>4:
            # todo:  process cluster
            cluster_id = int(records[4])
    else:
        line = [ln.split(",") for ln in line.split()]


    # filter the line/session shorter than 10
    if len(line) <= min_len:
        return [], [], [],[]

    # max seq len
    if seq_len is not None:
        line = line[:seq_len]
        token_labels = token_labels[:seq_len]

    if adaptive_window:
        window_size = len(line)

    line = np.array(line)
    token_labels = np.array(token_labels)

    # if time duration exists in data
    if len(line.shape)>1 and line.shape[1] == 2:
        tim = line[:,1].astype(float)
        line = line[:, 0]

        # the first time duration of a session should be 0, so max is window_size(mins) * 60
        tim[0] = 0
    else:
        line = line.squeeze()
        # if time duration doesn't exist, then create a zero array for time
        tim = np.zeros(line.shape)

    logkey_seqs = []
    time_seq = []
    seq_labels = []
    token_label_seq = []
    seq_ids = []
    for i in range(0, len(line), window_size):
        seq_ids.append(last_seq_id)
        logkey_seqs.append(line[i:i + window_size].tolist())
        time_seq.append(tim[i:i + window_size].tolist())
        seq_labels.append(np.amax(token_labels[i: i+window_size]))
        token_label_seq.append(token_labels[i:i+window_size].tolist())
        last_seq_id+=1

    return logkey_seqs, time_seq, seq_labels, token_label_seq, seq_ids

# WINDOW SIZE = 120

def generate_train_valid(data_path, window_size=20, adaptive_window=True,
                         sample_ratio=1, valid_size=0.1, output_path=None,
                         scale=None, scale_path=None, seq_len=None, min_len=0, is_label=False):
    with open(data_path, 'r') as f:
        data_iter = f.readlines()

    num_session = int(len(data_iter) * sample_ratio)
    # only even number of samples, or drop_last=True in DataLoader API
    # coz in parallel computing in CUDA, odd number of samples reports issue when merging the result
    # num_session += num_session % 2

    test_size = int(min(num_session, len(data_iter)) * valid_size)
    # only even number of samples
    # test_size += test_size % 2

    print("before filtering short session")
    print("train size ", int(num_session - test_size))
    print("valid size ", int(test_size))
    print("="*40)

    logkey_seq_pairs = []
    time_seq_pairs = []
    label_seq_paris = []
    token_label_seq_paris = []
    seq_id_paris = []
    session = 0
    last_seq_id = 0
    for line in tqdm(data_iter):
        if session >= num_session:
            break
        session += 1

        #generate multiple window sequences out of one block_id sequence
        logkeys, times, seq_label, token_labels, seq_ids = fixed_window(line, window_size, adaptive_window, seq_len, min_len,
                                                                        is_label= is_label,last_seq_id= last_seq_id)
        last_seq_id = max(seq_ids)+1
        logkey_seq_pairs += logkeys
        time_seq_pairs += times
        label_seq_paris +=seq_label
        token_label_seq_paris +=token_labels
        seq_id_paris += seq_ids

    logkey_seq_pairs = np.array(logkey_seq_pairs,  dtype=object)
    time_seq_pairs = np.array(time_seq_pairs,  dtype=object)
    label_seq_paris = np.array(label_seq_paris,  dtype=object)
    token_label_seq_paris = np.array(token_label_seq_paris,  dtype=object)
    seq_id_paris = np.array(seq_id_paris, dtype= object)


    logkey_trainset, logkey_validset, time_trainset, time_validset,\
    label_trainset, label_validset,\
        token_label_trainset, token_label_validset, seq_id_trainset, seq_id_validset = train_test_split(logkey_seq_pairs,
                                                      time_seq_pairs,
                                                      label_seq_paris,
                                                      token_label_seq_paris,
                                                      seq_id_paris,
                                                      test_size=test_size,
                                                      random_state=1234)

    # sort seq_pairs by seq len in descending order
    train_len = list(map(len, logkey_trainset))
    valid_len = list(map(len, logkey_validset))

    # the indices of training data in seq len descending order
    train_sort_index = np.argsort(-1 * np.array(train_len))
    valid_sort_index = np.argsort(-1 * np.array(valid_len))

    logkey_trainset = logkey_trainset[train_sort_index]
    logkey_validset = logkey_validset[valid_sort_index]

    time_trainset = time_trainset[train_sort_index]
    time_validset = time_validset[valid_sort_index]

    label_trainset = label_trainset[train_sort_index]
    label_validset = label_validset[valid_sort_index]

    token_label_trainset = token_label_trainset[train_sort_index]
    token_label_validset = token_label_validset[valid_sort_index]

    seq_id_trainset = seq_id_trainset[train_sort_index]
    seq_id_validset = seq_id_validset[valid_sort_index]

    train_df = pd.DataFrame()
    train_df["seq_id"] = seq_id_trainset
    train_df["logkey_seq"] = logkey_trainset
    train_df["time_seq"] = time_trainset
    train_df["seq_anomaly_label"] = label_trainset
    train_df["token_anomaly_label"] = token_label_trainset

    train_df.to_csv(f"{data_path}train.csv")

    valid_df = pd.DataFrame()
    valid_df["seq_id"] = seq_id_validset
    valid_df["logkey_seq"] = logkey_validset
    valid_df["time_seq"] = time_validset
    valid_df["seq_anomaly_label"] = label_validset
    valid_df["token_anomaly_label"] = token_label_validset
    valid_df.to_csv(f"{data_path}valid.csv")

    print("="*40)
    print("Num of train seqs", len(logkey_trainset))
    print("Num of valid seqs", len(logkey_validset))
    print("="*40)


    return logkey_trainset, logkey_validset, time_trainset, time_validset,label_trainset, label_validset,token_label_trainset, token_label_validset

def generate_test(data_path, data_file, window_size, adaptive_window, seq_len, scale, min_len):
    """
    :return: log_seqs: num_samples x session(seq)_length, tim_seqs: num_samples x session_length
    """
    log_seqs = []
    tim_seqs = []
    label_seq_paris = []
    token_label_seq_paris = []
    with open(data_path + data_file, "r") as f:
        for idx, line in enumerate(f.readlines()):
            #if idx > 40: break
            log_seq, tim_seq, labels, token_labels = fixed_window(line, window_size,
                                            adaptive_window=adaptive_window,
                                            seq_len=seq_len, min_len=min_len, is_label=True)
            if len(log_seq) == 0:
                continue

            # if scale is not None:
            #     times = tim_seq
            #     for i, tn in enumerate(times):
            #         tn = np.array(tn).reshape(-1, 1)
            #         times[i] = scale.transform(tn).reshape(-1).tolist()
            #     tim_seq = times

            log_seqs += log_seq
            tim_seqs += tim_seq
            label_seq_paris+= labels
            token_label_seq_paris += token_labels

    # sort seq_pairs by seq len
    log_seqs = np.array(log_seqs, dtype=object)
    tim_seqs = np.array(tim_seqs, dtype=object)
    label_seq_paris = np.array(label_seq_paris, dtype=object)
    token_label_seq_paris = np.array(token_label_seq_paris, dtype=object)

    test_len = list(map(len, log_seqs))
    test_sort_index = np.argsort(-1 * np.array(test_len))

    log_seqs = log_seqs[test_sort_index]
    tim_seqs = tim_seqs[test_sort_index]
    label_seq = label_seq_paris[test_sort_index]
    token_label_seq = token_label_seq_paris[test_sort_index]

    test_df = pd.DataFrame()
    test_df["logkey_seq"] = log_seqs
    test_df["time_seq"] = tim_seqs
    test_df["seq_anomaly_label"] = label_seq
    test_df["token_anomaly_label"] = token_label_seq
    test_df.to_csv(f"{data_path}{data_file}.csv")

    print(f"{data_file} size: {len(log_seqs)}")
    return log_seqs, tim_seqs, label_seq, token_label_seq

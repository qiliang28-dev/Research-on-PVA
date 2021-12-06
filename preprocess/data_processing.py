import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from NLFE.nlfe import poincare_plot, recurrence_plot, approximate_entropy, fuzzy_entropy, sample_entropy
from scipy import signal
from tqdm import tqdm

import file_list
import interaction
from features import FormattedData
from sklearn import preprocessing
import matplotlib.pyplot as plt


def getcwd():
    getcwd = os.getcwd()
    if 'preprocess' in getcwd:
        getcwd = getcwd.replace('preprocess', '')
    return getcwd


class DataSetMaker:
    def __init__(self, data_groups, output_path, output_name, fea_num=5):
        self.labeled_file_path = "/raw_data/original_data3/label"
        self.raw_file_path = "/raw_data/original_data3/seg_data"
        self.output_path = output_path
        self.output_name = output_name
        self.data_groups = data_groups
        self.window_size = 40
        self.stride = 40
        self.window_nums = fea_num

    def _next_file_data(self):
        path = getcwd() + self.labeled_file_path
        all_label_files = os.listdir(path)
        for seq_num in self.data_groups:
            labeled_file_name = "{}_label.csv".format(seq_num)
            if labeled_file_name in all_label_files:
                labeled_data = np.loadtxt(os.path.join(path, labeled_file_name), dtype=int,
                                          delimiter=",")
                yield seq_num, labeled_data
            else:
                raise ValueError("No such file {}, Please check it name!".format(labeled_file_name))

    def generate_sample(self, is_normalization):
        features = []
        labels = []
        desc = {}
        for seq_num, labeled_data in self._next_file_data():
            label = labeled_data[:, 1]
            for i in tqdm(range(labeled_data.shape[0]), desc="raw file {}".format(seq_num)):
                if label[i] == -1:
                    continue
                file_seq = labeled_data[i, 0]
                file_name = os.path.join(os.path.join(self.raw_file_path, str(seq_num)), "{}.csv".format(file_seq))
                raw_data = np.loadtxt(getcwd() + file_name, dtype=float, delimiter=",")
                if is_normalization:
                    raw_data = preprocessing.scale(raw_data)
                raw_data = np.transpose(raw_data)
                padding_nums = 100 - raw_data.shape[1]
                if not padding_nums % 2:
                    l_padding = int(padding_nums / 2)
                    padding_t = (l_padding, l_padding)
                else:
                    l_padding = int((padding_nums - 1) / 2)
                    padding_t = (l_padding, padding_nums - l_padding)

                raw_data = np.pad(raw_data, ((0, 0), padding_t), "edge")
                features.append(raw_data.flatten())
                labels.append(label[i])

        formatted_data = FormattedData(np.array(features), np.array(labels), desc)
        with open(os.path.join(self.output_path, "{}.pkl".format(self.output_name)), "wb") as f:
            pickle.dump(formatted_data, f)

    def _calc_en_fea(self, data, entropy_func, n, m=3, r_coe=1):
        data_len = data.shape[0]
        cols_num = data.shape[1]
        # samples = []
        self.window_size = self.stride = int(data_len / self.window_nums)
        # window_nums = int((data_len - self.stride) / self.window_size + 1)
        file_fea = np.zeros((self.window_nums, cols_num))
        r = np.std(data, axis=0)
        break_flag = False
        for j in range(cols_num):
            for i in range(self.window_nums):
                s = i * self.stride
                e = s + self.window_size
                ts = data[s:e, :]
                r = r_coe * np.std(ts, axis=0)
                if len(ts[:, j]) > m + 1:
                    if not n:
                        file_fea[i, j] = entropy_func(ts[:, j], m, r[j])
                    else:
                        file_fea[i, j] = entropy_func(ts[:, j], m, n, r_coe)
                else:
                    break_flag = True
                    break
            if break_flag:
                break
        return np.transpose(file_fea).flatten(), break_flag

    def _calc_non_en_fea(self, data, nlfe_func, **kwargs):
        data_len = data.shape[0]
        cols_num = data.shape[1]
        self.window_size = self.stride = int(data_len / self.window_nums)
        # window_nums = int((data_len - self.stride) / self.window_size + 1)
        file_fea = np.zeros((self.window_nums * 3, cols_num))
        # samples = []
        for j in range(cols_num):
            for i in range(self.window_nums):
                s = i * self.stride
                e = s + self.window_size
                ts = data[s:e, :]
                func_map = {
                    "poincare": poincare_plot,
                    "recurrence": recurrence_plot,
                }
                if nlfe_func == "poincare":
                    sd1, sd2, ratio = func_map[nlfe_func](ts)
                    file_fea[i * 3, j] = sd1
                    file_fea[i * 3 + 1, j] = sd1
                    file_fea[i * 3 + 2, j] = ratio
                else:
                    RP, REC = func_map.get(nlfe_func, Exception)(ts, **kwargs)
                    file_fea[i, j] = REC
        return np.transpose(file_fea).flatten(), False

    def _standardized_data(self, data):
        for i in range(data.shape[1]):
            pass

    def extract_fea(self, entropy_func, n=0, m=3, r_coe=1, is_normalization=False):
        entropy_func_map = dict(
            approximate_en=approximate_entropy,
            sample_en=sample_entropy,
            fuzzy_en=fuzzy_entropy,
        )
        features = []
        labels = []
        desc = {}
        for seq_num, labeled_data in self._next_file_data():
            label = labeled_data[:, 1]
            for i in tqdm(range(labeled_data.shape[0]), desc="raw file {}".format(seq_num)):
                if label[i] == -1:
                    continue
                file_seq = labeled_data[i, 0]
                file_name = os.path.join(os.path.join(self.raw_file_path, str(seq_num)), "{}.csv".format(file_seq))
                raw_data = np.loadtxt(getcwd() + file_name, dtype=float, delimiter=",")
                # 数据标准化
                if is_normalization:
                    raw_data = preprocessing.scale(raw_data)
                if str(entropy_func).lower().endswith("en"):
                    feature, break_flag = self._calc_en_fea(raw_data,
                                                            entropy_func_map.get(str(entropy_func).lower(), Exception),
                                                            n,
                                                            m,
                                                            r_coe)
                else:
                    feature, break_flag = self._calc_non_en_fea(raw_data, entropy_func)
                if not break_flag:
                    features.append(feature)
                    labels.append(label[i])
                else:
                    continue

        # print(features)
        formatted_data = FormattedData(np.array(features), np.array(labels), desc)
        with open(os.path.join(self.output_path, "{}.pkl".format(self.output_name)), "wb") as f:
            pickle.dump(formatted_data, f)


class SegData:

    def __init__(self, seq_num, rr=None):
        self.seq_num = seq_num
        self.file_name = "{}.csv".format(self.seq_num)
        self.raw_data_path = "../../../temp/code/raw_data/original_data3/"
        self.interested_data_path = "../../../temp/code/raw_data/original_data3/interested_data"
        self.seged_data_path = "../raw_data/original_data3/seg_data/{}".format(seq_num)
        self.seged_data_by_type_path = "../raw_data/original_data3/seg_by_type/{}".format(seq_num)
        self.interest_cols = [1, 2, 4]
        self.temp_index = []

        self.rr = rr  # unit cpm
        self.fs = 100  # expected sampling frequency is 100Hz
        self.win_size = int(self.fs * 60 / self.rr)
        self.win_limit = int(self.win_size * 2 / 3)
        self.interest_data = None
        # self.rr = [17, 21, 21]  # unit cpm

    def _get_df_raw_data(self):
        """ Acquiring raw data from csv file. """
        return pd.read_csv(os.path.join(self.raw_data_path, self.file_name))

    def calc_real_fs(self):
        """ Calculating the real mean sampling frequency. """
        raw_data_df = self._get_df_raw_data()
        df_gb = raw_data_df.groupby('Time ')
        temp = []
        for index, data in df_gb:
            data_np = data.values
            temp.append(data_np.shape[0])
        self.fs = int(np.mean(np.array(temp[1:-1])))
        print("The real mean fs of file {}: {}Hz".format(self.seq_num, self.fs))

    def update_win_size(self):
        """ Estimating the real size of one window of a breath cycle according to preconfigured respiratory rate. """
        self.win_size = int(self.fs * 60 / self.rr)
        print("The win_size of file {} : {}".format(self.seq_num, self.win_size))

    def low_pass_filter(self, fcs, data):
        for i in range(3):
            wn = 2 * fcs[i] / self.fs
            b, a = signal.butter(3, wn, 'lowpass')
            data[:, i] = signal.filtfilt(b, a, data[:, i])
        return data

    def get_index4(self, dependency="flow", back_point=1):
        """ Segmenting data based on differential value."""
        interest_col = 0
        if dependency is "flow":
            pass
        elif dependency is "tidal":
            interest_col = 1
        else:
            # pressure
            interest_col = 2
        raw_data_df = self._get_df_raw_data()
        raw_data_np = raw_data_df.values
        self.interest_data = raw_data_np[:, self.interest_cols]
        if not Path(self.interested_data_path).exists():
            os.mkdir(self.interested_data_path)
        np.savetxt(os.path.join(self.interested_data_path, "{}.csv".format(self.seq_num)), self.interest_data,
                   delimiter=",")

        interest_column = self.interest_data[:, interest_col]
        d_col = interest_column[1:] - interest_column[:-1]
        first_start_index = np.argmax(d_col[:int(self.win_size * 3 / 2)])
        if first_start_index - back_point >= 0:
            self.temp_index.append(first_start_index - back_point)
        else:
            self.temp_index.append(first_start_index)
        abs_start_index = first_start_index
        while True:
            last_start_index = abs_start_index + (2 * self.fs)
            rel_next_start_index = np.argmax(d_col[last_start_index:last_start_index + self.win_size])
            abs_start_index = last_start_index + rel_next_start_index
            self.temp_index.append(abs_start_index - back_point)
            if abs_start_index + self.win_size + self.fs > np.size(d_col):
                break
        print(self.temp_index)

    def get_index3(self):
        """ Segmenting data according to the peak value of volume of each cycle. """
        raw_data_df = self._get_df_raw_data()
        raw_data_np = raw_data_df.values
        self.interest_data = raw_data_np[:, self.interest_cols]
        if not Path(self.interested_data_path).exists():
            os.mkdir(self.interested_data_path)
        np.savetxt(os.path.join(self.interested_data_path, "{}.csv".format(self.seq_num)), self.interest_data,
                   delimiter=",")

        tidal = self.interest_data[:, 1]
        stride_size = int(self.win_size * 3 / 2)
        first_peak_index = np.argmax(tidal[:stride_size])
        abs_next_peak_index = first_peak_index
        if first_peak_index >= int(self.win_size / 2):
            self.temp_index.append(first_peak_index - int(self.win_size / 2))
        # self.temp_index.append(first_peak_index)
        while True:
            last_peak_index = abs_next_peak_index + self.fs
            rel_next_peak_index = np.argmax(tidal[last_peak_index:last_peak_index + stride_size])
            abs_next_peak_index = last_peak_index + rel_next_peak_index
            start_index = abs_next_peak_index - int(self.win_size / 2)
            self.temp_index.append(start_index)
            if abs_next_peak_index + stride_size > np.size(tidal):
                if np.size(tidal) - abs_next_peak_index > int(self.win_size / 2):
                    self.temp_index.append(abs_next_peak_index + int(self.win_size / 2))
                break
        # print(self.temp_index)

    def get_index2(self, PEEP=5):
        """ 根据压力波形在吸气触发时会有一个压力下陷，此时的压力值低于设置的PEEP值，因此PEEP值需要作为参数传入"""
        raw_data_df = self._get_df_raw_data()
        raw_data_np = raw_data_df.values
        self.interest_data = raw_data_np[:, self.interest_cols]

  
        if not Path(self.interested_data_path).exists():
            os.mkdir(self.interested_data_path)
        np.savetxt(os.path.join(self.interested_data_path, "{}.csv".format(self.seq_num)), self.interest_data,
                   delimiter=",")
        Paw = self.interest_data[:, 2]
        mask_paw = np.where(Paw >= PEEP, 1, 0)
        d_mask = mask_paw[:-1] - mask_paw[1:]
        self.temp_index = np.argwhere(d_mask == 1)
        self.temp_index = self.temp_index.reshape((1, -1))[0]
        print(self.temp_index)

    def get_index(self):
        """ 根据相应的窗口大小，以及每个窗口内的最小值来得到一个窗口内的起始位置索引 """
        raw_data_df = self._get_df_raw_data()
        raw_data_np = raw_data_df.values
        self.interest_data = raw_data_np[:, self.interest_cols]
        # Filtering algorithm is needed if the sampleing frequency is 100 Hz.
        # fcs = []
        # num = [10, 20, 20]
        # for i in range(3):
        #     fcs1 = filtering.fftspectrum(self.interest_data[:, i], self.fs, num[i])
        #     fcs.append(fcs1)
        # self.interest_data = self.low_pass_filter(fcs, self.interest_data)
        if not Path(self.interested_data_path).exists():
            os.mkdir(self.interested_data_path)
        np.savetxt(os.path.join(self.interested_data_path, "{}.csv".format(self.seq_num)), self.interest_data,
                   delimiter=",")
        tidal = self.interest_data[:, 1]
        # print(interest_data[:10, :])
        start_index = 0
        self.temp_index.append(start_index)
        while True:
            s = start_index + self.fs
            e = s + self.win_size
            if s >= tidal.shape[0] - 1:
                break
            if e >= tidal.shape[0]:
                e = tidal.shape[0] - 1

            temp = np.argmin(tidal[s:e])
            # print(s, e, tidal.shape[0], temp)
            next_index = s + temp
            self.temp_index.append(next_index)
            start_index = next_index

        print("have got index .....")
        # print(self.temp_index[:10])

    def save_data(self):
        """ 将分割好的每个周期的数据按文件保存 """
        for i in range(len(self.temp_index) - 1):
            cur_data = self.interest_data[self.temp_index[i]:self.temp_index[i + 1], :]
            if not Path(self.seged_data_path).exists():
                os.mkdir(self.seged_data_path)

            np.savetxt(os.path.join(self.seged_data_path, "{}.csv".format(i + 1)), cur_data, delimiter=",")
        print("file {}.csv finished".format(self.seq_num))

    def archive_file(self, ):
        """ 根据归类好的列表 再次将各文件移到新的文件夹 """
        dir_i = [file_list.dir_1]
        for d_i in dir_i:
            for k, v in d_i.items():
                for f_name in v:
                    shutil.move(os.path.join(self.seged_data_path, f_name),
                                os.path.join(self.seged_data_by_type_path, k))

    def start(self):
        self.calc_real_fs()
        self.update_win_size()
        # self.get_index4(dependency="tidal", back_point=10)
        # self.get_index4(dependency="flow", back_point=0)
        # self.get_index4(dependency="pressure", back_point=3)
        self.get_index()
        self.save_data()


def start_extract_fea(is_normalization=False, fea_name="approximate_en", n=0, m=2, r_coe=2.5):
    """ """
    # 提取特征,各组分别包含了相应的文件夹
    group1 = [1, 2, 3, 4, 5, 29, 30, 31, 32, 33, 34, 35, 47, 48, 49, 50]
    group2 = [6, 7, 8, 37, 46, 60]
    group3 = [9, 10, 12, 13, 14, 15, 38, 39, 40, 41, 42, 43, 44, 45, 51, 52, 53, 54, 55, 56, 57]
    group4 = [16, 17, 18, 19, 20, 21, 22]
    group5 = [28, 63]
    group6 = [11, 25, 26, 27, 61, 62, 64]
    groups = [group1, group2, group3, group4, group5, group6]
    out_path = {
        "approximate_en": getcwd() + "/features/approximate_entropy_feature",
        "sample_en": getcwd() + "/features/sample_entropy_feature",
        "fuzzy_en": getcwd() + "/features/fuzzy_entropy_feature",
        "poincare": getcwd() + "/features/poincare_feature",
        "raw": getcwd() + "/features/raw",
    }
    for i in range(len(groups)):
        out_name = str(i + 1)
        dataset_maker = DataSetMaker(groups[i], out_path.get(fea_name, Exception), out_name, fea_num=4)
        if fea_name != "raw":
            dataset_maker.extract_fea(fea_name, n, m, r_coe, is_normalization=is_normalization)
        else:
            dataset_maker.generate_sample(is_normalization=is_normalization)


def start_pre_process(start_seq, end_seq):
    rr = [12, 12, 12, 12, 12,  # 1
          35, 24, 22,  # 6
          22, 22, 10, 22, 22, 23, 23,  # 9
          20, 20, 25, 20, 20, 20, 20,  # 16
          22, 20, 20, 10, 20, 20,  # 23
          18, 18, 18, 18,  # 29
          30, 30, 30, 32, 20,  # 33
          23, 24, 23, 12, 23, 23,  # 38
          15, 16, 24,  # 44
          18, 18, 18,  # 47
          30, 23, 23, 12, 23, 23,  # 50
          15, 16, 23,  # 56
          19, 24, 20, 10, 20, 20  # 59
          ]
    for index in range(start_seq, end_seq + 1):
        segmenter = SegData(index, rr[index - 1])
        segmenter.start()

        inter_tag = interaction.InterTag(index)
        inter_tag.save_pics()


if __name__ == '__main__':
    
    # data segmentation 
    start_pre_process(62, 62)
    
    """
    # extract features automatically
    start_extract_fea(is_normalization=True, fea_name="poincare", n=0, m=0, r_coe=0)
    """
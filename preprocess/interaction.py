import matplotlib.pyplot as plt
import matplotlib.image as implt
import pandas as pd
import numpy as np

import os
import pickle
from pathlib import Path
from tqdm import tqdm
import time


class InterTag:
    def __init__(self, seq_num):
        self.data_path = "../raw_data/original_data3/seg_data/"
        self.seq_num = seq_num
        self.pics_path = "../raw_data/original_data3/pics/{}".format(seq_num)
        self.outcome_data_path = "../raw_data/original_data3/seg_data/label_pkl/{}".format(seq_num)
        # self.sub_dir = [1, 2, 3]
        self.asy_map = {
            "1": "DT",  # double trigger
            "2": "IEE",  # ineffective effort
            "3": "Other",  # other type
            "4": "PC",  # premature cycling
            "5": "DC",  # delayed cycling
        }
        self.sub_dir = ['Other']
        self.tag_list = []

    def _next_file_data(self):
        sub_path = os.path.join(self.data_path, str(self.seq_num))
        all_file = os.listdir(sub_path)
        for file_name in all_file:
            data = np.loadtxt(os.path.join(sub_path, file_name), delimiter=",")
            yield file_name, data

    def save_pics(self):
        labels = ["flow", "tidal", "Paw"]
        cols = [2, 0, 1]
        for file_name, data in tqdm(self._next_file_data(), desc="Processing of file {}.csv".format(self.seq_num),
                                    leave=True, ncols=100, unit='B', unit_scale=True):
            fig, ax = plt.subplots(3, 1, clear=True, figsize=(8, 6), sharex=True)
            for i in range(3):
                axi = ax[i]
                axi.plot(data[:, cols[i]], label=labels[cols[i]])
                axi.legend(loc="upper right")
                axi.legend(loc="upper right")
            fig.suptitle(file_name)
            if not Path(self.pics_path).exists():
                os.mkdir(self.pics_path)
            plt.savefig(os.path.join(self.pics_path, file_name.split(".")[0]))
            plt.close()

    def start(self):
        all_img = os.listdir(self.pics_path)
        for img_name in all_img:
            x = implt.imread(os.path.join(self.pics_path, img_name))
            plt.imshow(x)
            plt.axis('off')
            plt.show()
            while True:
                try:
                    tag = eval(input("please input your judge: "))
                    print("{} file successfully tagged {}".format(img_name, tag))
                    self.tag_list.append([int(img_name.split(".")[0]), tag])
                    plt.close()
                    break
                except Exception as e:
                    print(e)
                    print(img_name)
            # if self.tag_list:
            # with open(os.path.join(self.outcome_data_path, "{}_labeled_data.pkl".format(self.seq_num)), "wb") as f:
            #     pickle.dump(self.tag_list, f)
            # print("Finished. Successfully saved file into {}".format(self.outcome_data_path))


def temp():
    with open(os.path.join("../raw_data/original_data/seg_data/label_pkl", "1_labeled_data.pkl"), "rb") as f:
        tag_list = pickle.load(f)
        tag_list = np.array(tag_list)
        dt_list = tag_list[:, 0][np.argwhere(tag_list[:, 1] == 1)]
        iee_list = tag_list[:, 0][np.argwhere(tag_list[:, 1] == 2)]
        other_list = tag_list[:, 0][np.argwhere(tag_list[:, 1] == 3)]
        dt_list = list(np.reshape(dt_list, (dt_list.shape[0],)))
        iee_list = list(np.reshape(iee_list, (iee_list.shape[0],)))
        other_list = list(np.reshape(other_list, (other_list.shape[0],)))
        for l in [dt_list, iee_list, other_list]:
            t = []
            for n in l:
                t.append(str(n) + ".csv")
            print(t)


if __name__ == '__main__':
    # temp()
    for i in range(23, 24):
        inter_tag = InterTag(i)
        # inter_tag.start()
        inter_tag.save_pics()

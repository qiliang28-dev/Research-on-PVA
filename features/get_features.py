import pickle
import os
import sys

"""this imported package can not be deleted
Only with it could pickle call function of load
because of the FormattedData defined by myself."""
from features import FormattedData

__all__ = ["ds"]


class DataSets:
    def __init__(self, ):
        self.base_path = "../features"
        self.fea_dir_map = dict(fuzzy="fuzzy_entropy_feature",
                                approximate="approximate_entropy_feature",
                                poincare="poincare_feature",
                                sample="sample_entropy_feature",
                                raw="raw")

    def get_data(self, fea_type="fuzzy", group=1):
        data_path = os.path.join(self.base_path, self.fea_dir_map[fea_type])
        # all_file = os.listdir(data_path)
        # sys.stdout.write("you choose feature of {} as sample\n".format(fea_type))
        # sys.stdout.close()
        # print(os.path.join("../features", self._fea_name_map[fea_name]))
        with open(os.path.join(data_path, "{}.pkl".format(str(group))), "rb") as pkl_file:
            data = pickle.load(pkl_file)
        return data


def get_fea():
    sample_pkl_file = open("../features/samples.pkl", "rb")
    label_pkl_file = open("../features/labels.pkl", "rb")

    samples = pickle.load(open("../features/samples.pkl", "rb"))
    labels = pickle.load(open("../features/labels.pkl", "rb"))

    sample_pkl_file.close()
    label_pkl_file.close()
    return samples, labels


ds = DataSets()
if __name__ == '__main__':
    dataset = ds.get_data("sample")
    print(type(dataset.data))
    print(type(dataset.label))
    pass

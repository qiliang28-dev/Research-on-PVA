import os
import random
import warnings

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import naive_bayes
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.metrics import (confusion_matrix, precision_score,
                             classification_report, cohen_kappa_score, roc_auc_score, auc)
from sklearn.exceptions import ConvergenceWarning
from tensorflow import keras
from tensorflow_core.python.keras import layers
from tensorflow_core.python.keras.callbacks import EarlyStopping
from xgboost import XGBClassifier
import tensorflow as tf

from features import FormattedData
# from sklearn.utils import shuffle
# import pickle

# from features import get_features
from features.get_features import ds

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from pathlib import Path


def split_dataset(samples, labels, test_size=0.2, random_state=666):
    x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=test_size,
                                                        shuffle=True, random_state=random_state)
    return x_train, x_test, y_train, y_test


def draw_auc_curve(y_true, y_prob):
    """
    Just process problem of binary classification
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true.ravel(), y_prob.ravel())
    auc = metrics.auc(fpr, tpr)
    print(fpr, tpr, thresholds)
    print(auc)
    lw = 2
    plt.figure()  # figsize=(10,10)
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve(AUC=%0.4f)' % auc)  # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


def draw_confusion_matrix(cm, label_names, path, name):
    sns.set()
    mpl.rcParams['font.sans-serif'] = 'Times New Roman'
    mpl.rcParams['axes.unicode_minus'] = False
    f, ax = plt.subplots()
    sns.heatmap(cm, annot=True, ax=ax, fmt="d")
    ax.set(xticklabels=label_names, yticklabels=label_names)
    ax.set_title('Confusion Matrix of {}'.format(name), fontsize=16, fontweight='bold')
    ax.set_xlabel('Pred Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(os.path.join(path, name), name))
    # TODO
    # plt.show()


def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return np.argmax(b, axis=1)


def evaluate_report(clf, x_test, y_test, path, name, pva_class, pva_label):
    y_pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)
    if len(y_pred.shape) > 1:
        y_pred = props_to_onehot(y_pred)
    cf_matrix = confusion_matrix(y_test, y_pred)

    draw_confusion_matrix(cf_matrix, pva_class, path, name)
    print("confusion matrix: \n", cf_matrix)
    report = "confusion matrix: \n{}\n\n".format(cf_matrix)
    micro_precision_s = precision_score(y_test, y_pred, average="micro")
    macro_precision_s = precision_score(y_test, y_pred, average="macro")
    weight_precision_s = precision_score(y_test, y_pred, average="weighted")

    print(
        "micro precision score: {}\nmacro precision score: {}\nweighted precision score: {}\n".format(micro_precision_s,
                                                                                                      macro_precision_s,
                                                                                                      weight_precision_s))

    report += "micro precision score: {}\nmacro precision score: {}\nweighted precision score: {}\n".format(
        micro_precision_s,
        macro_precision_s,
        weight_precision_s)

    classification_report_rs = classification_report(y_test, y_pred, labels=np.array(pva_label),
                                                     target_names=np.array(pva_class), digits=5)

    print("classification report:\n", classification_report_rs)
    report += classification_report_rs

    cohen_kappa_s = cohen_kappa_score(y_test, y_pred)
    print("\ncohen kappa score: ", cohen_kappa_s)
    report += "\ncohen kappa score:{}".format(cohen_kappa_s)

    roc_score = roc_auc_score(y_test, y_prob, multi_class="ovo")

    print("\nroc-ovo: ", roc_score)
    report += "\nroc-ovo:{}".format(roc_score)
    return (report, weight_precision_s, [micro_precision_s, macro_precision_s, weight_precision_s, cohen_kappa_s])


def save_results(path, name, content):
    with open(os.path.join(os.path.join(path, name), "{}.txt".format(name)), "w+") as f:
        f.write(content)


def get_dataset(fea_type, groups):
    num = len(groups)
    i = 0
    old_sample = None
    old_label = None
    while i < num:
        seq = groups[i]
        dataset_i = ds.get_data(fea_type, group=seq)
        samples_i, labels_i = dataset_i.data, dataset_i.label
        if not i:
            old_sample = np.copy(samples_i)
            old_label = np.copy(labels_i)
            i += 1
            continue
        else:
            old_sample = np.r_[old_sample, samples_i]
            old_label = np.r_[old_label, labels_i]
        i += 1
    return old_sample, old_label


def logistic_regression(samples, labels, test_size=0.2, random_state=666):
    x_train, x_test, y_train, y_test = split_dataset(samples, labels, test_size=test_size, random_state=random_state)
    # 默认是OVR
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    return clf, x_train, y_train, x_test, y_test


def nb(samples, labels, test_size=0.2, random_state=666):
    x_train, x_test, y_train, y_test = split_dataset(samples, labels, test_size=test_size, random_state=random_state)
    clf = naive_bayes.GaussianNB()
    clf.fit(x_train, y_train)
    return clf, x_train, y_train, x_test, y_test


def svm(samples, labels, test_size=0.2, random_state=666):
    x_train, x_test, y_train, y_test = split_dataset(samples, labels, test_size=test_size, random_state=random_state)
    clf = SVC(C=1, kernel="rbf", gamma=1, decision_function_shape='ovo', probability=True, verbose=1)
    clf.fit(x_train, y_train)
    # pred = clf.predict(x_test)
    # pred_prob = clf.predict_proba(x_test)
    # get_metric(y_test, pred, pred_prob, "../results", "SVM")
    return clf, x_train, y_train, x_test, y_test


def random_forest(samples, labels, test_size=0.2, random_state=666):
    x_train, x_test, y_train, y_test = split_dataset(samples, labels, test_size=test_size, random_state=random_state)
    clf = RandomForestClassifier(verbose=0)
    clf.fit(x_train, y_train)
    return clf, x_train, y_train, x_test, y_test


def voting(samples, labels, test_size=0.2, random_state=666):
    x_train, x_test, y_train, y_test = split_dataset(samples, labels, test_size=test_size, random_state=random_state)

    clf = VotingClassifier(estimators=[('rf_clf', RandomForestClassifier(verbose=0)), ('xgb', XGBClassifier())],
                           voting='soft')
    # return clf, X, X_predict, y, y_predict
    clf.fit(x_train, y_train)
    return clf, x_train, y_train, x_test, y_test


def cnn(samples, labels, test_size=0.2, random_state=666):
    x_train, x_test, y_train, y_test = split_dataset(samples, labels, test_size=test_size, random_state=random_state)
    x_train, x_val, y_train, y_val = split_dataset(x_train, y_train, test_size=test_size, random_state=random_state)

    model = keras.Sequential()

    x_train = np.expand_dims(x_train, axis=2)
    x_val = np.expand_dims(x_val, axis=2)
    x_test = np.expand_dims(x_test, axis=2)

    kernel_size = 6
    stride = 5
    epoch = 15
    model.add(layers.Conv1D(kernel_size, stride, padding='same', input_shape=(300, 1)))
    # model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(kernel_size, stride, padding='same'))
    # model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    # model.add(layers.Conv1D(4, 2, padding='same'))
    model.add(layers.Conv1D(kernel_size, stride, padding='same'))
    # model.add(layers.MaxPool1D())
    model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation='relu', input_dim=x_train.shape[1]))
    model.add(layers.Dense(512, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dense(256, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dense(128, activation='relu', kernel_initializer='normal'))
    # 分类层
    model.add(layers.Dense(64, activation='relu', kernel_initializer='normal'))
    model.add(layers.Dense(5, activation='softmax'))
    # back1 = EarlyStopping(monitor="val_accuracy", patience=300, verbose=0, mode='max')
    model.compile(optimizer=keras.optimizers.Adam(),
                  # loss=keras.losses.CategoricalCrossentropy(),  # 需要使用to_categorical
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.build()
    model.summary()

    # for i in range(epoch):
    history = model.fit(x_train, y_train, batch_size=32, epochs=epoch, validation_data=(x_val, y_val))
    # if history.history['val_accuracy'][0] > 0.976:
    #     break
    # res = model.evaluate(x_test, y_test)
    # print(res)
    _, train_acc = model.evaluate(x_train, y_train, verbose=0)
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print('Train: %.6f, Test: %.6f' % (train_acc, test_acc))
    # plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    # plt accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.legend()
    plt.show()
    return model, x_train, y_train, x_test, y_test


def cnn2(samples, labels, test_size=0.2, random_state=666):
    x_train, testX, y_train, testy = split_dataset(samples, labels, test_size=test_size, random_state=random_state)
    trainX, valX, trainy, valy = split_dataset(x_train, y_train, test_size=test_size, random_state=random_state)

    # one-hot labels
    trainy = keras.utils.to_categorical(trainy, num_classes=5)
    valy = keras.utils.to_categorical(valy, num_classes=5)
    testy = keras.utils.to_categorical(testy, num_classes=5)

    trainX = np.expand_dims(trainX, axis=2)
    valX = np.expand_dims(valX, axis=2)
    testX = np.expand_dims(testX, axis=2)

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(300, 1)))
    model.add(layers.Conv1D(filters=64, kernel_size=11, strides=1, padding='same', use_bias=True))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=(2), strides=1, padding='same'))
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv1D(32, kernel_size=5, strides=1, padding='same', use_bias=True, dilation_rate=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv1D(32, kernel_size=5, strides=1, padding='same', use_bias=True, dilation_rate=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv1D(32, kernel_size=3, strides=1, padding='same', use_bias=True, dilation_rate=4))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv1D(16, kernel_size=3, strides=1, padding='same', use_bias=True, dilation_rate=4))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=(2), strides=1, padding='same'))
    model.add(layers.Dropout(0.1))

    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(5, activation='softmax', use_bias=False))

    # compile model
    opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.build()
    model.summary()
    # fit model
    history = model.fit(trainX, trainy, validation_data=(valX, valy), batch_size=32, epochs=100, verbose=1)
    # evaluate the model
    _, train_acc = model.evaluate(trainX, trainy, verbose=1)
    _, test_acc = model.evaluate(testX, testy, verbose=1)
    print('Train: %.6f, Test: %.6f' % (train_acc, test_acc))
    # plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    # plt accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.legend()
    plt.show()
    return model, trainX, trainy, testX, testy


def xgboost(samples, labels, test_size=0.2, random_state=666):
    x_train, x_test, y_train, y_test = split_dataset(samples, labels, test_size=test_size, random_state=random_state)

    xgb_classifier = XGBClassifier()
    # kf = KFold(n_splits=
    xgb_classifier.fit(x_train, y_train)
    # print(clf.best_params_)

    # clf.fit(x_train, y_train)
    # print(clf.best_params_, clf.best_score_)
    return xgb_classifier, x_train, y_train, x_test, y_test


class MyModel:
    """可以根据输入的字符确定选用的模型以及特征， 同时输出保存具有相应特征名称的结果 """

    def __init__(self, model_name, fea_name, groups):
        self.model_name = str(model_name).lower()
        self.fea_name = str(fea_name).lower()
        self.groups = groups
        self.clf = None
        self.samples = None
        self.labels = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.pva_class = ["DT", "IEE", "Other", "PC", "DC"]
        self.pva_label = [0, 1, 2, 3, 4]
        self.output_path = os.getcwd() + "/results"
        self.output_name = "{}_{}".format(model_name, fea_name)
        self.report_content = ""
        if 'models/result' in self.output_path:
            self.output_path = self.output_path.replace('models/result', 'result')
        if not Path(os.path.join(self.output_path, self.output_name)).exists():
            os.mkdir(os.path.join(self.output_path, self.output_name))

    def get_dataset(self):
        self.samples, self.labels = get_dataset(self.fea_name, self.groups)
        return
        # 平衡各类样本数量达到一致
        num_min = 100000
        for i in range(5):
            num_i = np.where(self.labels == i)[0].shape[0]
            num_min = num_i if num_i < num_min else num_min
        temp_samples = np.zeros((num_min * 5, self.samples.shape[1]))
        temp_labels = np.zeros((num_min * 5,))
        for i in range(5):
            sample_i_index = np.reshape(np.argwhere(self.labels == i), (1, -1))[0]

            if sample_i_index.shape[0] > num_min:
                # temp_list = [ind for ind in range(sample_i_index.shape[0])]
                # 设置随机种子，保证效果一致，实际训练需要注释
                random.seed(10)
                random_i_index = random.sample(list(sample_i_index), num_min)
                sample_i = self.samples[random_i_index, :]
                label_i = self.labels[random_i_index]
            else:
                sample_i = self.samples[sample_i_index, :]
                label_i = self.labels[sample_i_index]
            s = i * num_min
            e = s + num_min
            temp_samples[s:e, :] = sample_i
            temp_labels[s:e] = label_i

        self.samples = temp_samples
        self.labels = temp_labels

    def get_model(self, test_size=0.2, random_state=666):
        model_map = dict(
            rf=random_forest,
            lr=logistic_regression,
            svm=svm,
            xgb=xgboost,
            cnn=cnn,
            cnn2=cnn2,
            voting=voting
        )
        self.clf, self.x_train, self.y_train, self.x_test, self.y_test = model_map.get(self.model_name, Exception)(
            self.samples, self.labels, test_size, random_state)

    def evaluate_model(self):
        report = evaluate_report(self.clf, self.x_test, self.y_test, self.output_path, self.output_name,
                                 self.pva_class,
                                 self.pva_label)
        self.report_content = report[0]
        return report[1], report[2]

    def save_rs(self):
        save_results(self.output_path, self.output_name, self.report_content)

    def start(self, test_size=0.2, random_state=666):
        self.get_dataset()
        # print(self.samples.shape)
        self.get_model(test_size=test_size, random_state=random_state)
        acc, indexes = self.evaluate_model()
        self.save_rs()
        return acc, indexes


if __name__ == '__main__':
    """
    rf: Random forest
    svm: support vector machine
    lr: logistic regression
    cnn: cnn
    xgb: XGBOOST
    voting: voting  
    """
    """
    approximate: approximate entropy  l
    fuzzy: fuzzy entropy  m fuzzy_en m4 n3 r=2.5
    sample: sample entropy z
    poincare:
    """
    # lst = []
    final_rs = []
    random_states = [s for s in range(666, 681)]
    for random_state in random_states:
        res = []
        resi = [random_state for _ in range(4)]
        res.append(resi)
        model_list = [
            "lr",
            "rf",
            "svm",
            "xgb",
            "voting"
        ]
        fea_list = [
            "approximate",
            "fuzzy",
            "sample",
            "poincare"
        ]
        for model in model_list:
            for fea in fea_list:
                my_model = MyModel(model_name=model, fea_name=fea, groups=[1, 2, 3, 4, 5, 6])
                _, indexes = my_model.start(test_size=0.3, random_state=random_state)
                res.append(indexes)

        res = np.array(res).reshape((-1, 4))
        final_rs.append(res)
    final_rs = np.array(final_rs).reshape((-1, 4))
    np.savetxt("../results/final_rs.csv", final_rs, delimiter=",")
    """
    my_model = MyModel(model_name="cnn2", fea_name="raw", groups=[1, 2, 3, 4, 5, 6])
    my_model.start()
    """

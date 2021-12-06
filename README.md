# Research-on-PVA

Research on patient ventilator asynchrony: a novel  respiratory dataset and four non-linear feature  extraction methods.

This file introduces the method that how to use the repo.

All the package dependencies were listed in requirements.txt. 

The features extracted from raw data have been uploaded for the convenient to skip front 4 steps to quickly get the results.

The novel respiratory dataset was zipped in a file dataset.7z which is of size beyond 20MB, whereas it will nearly occupy 90
MB of memory after releasing it, and the description about it ,including overview.xlsx,was deposited in ./dataset/description/.

### Unzip dataset
1. The file ./dataset/raw_data.7z should initially be unzipped to the directory ./raw_data/original_data3/

### Data Segmentation
1. Open data_processing.py file, and focus your eyes on the entrance of main function, in which there are two functions 
   included, namely uncommented function of start_pre_process() and commented function of start_extract_fea(). The two 
   functions can not uncommented at the same time as shown in following code. The function start_pre_process is used to 
   preprocess the dataset as you see in the comment, whose argument represents various groups that you want to segment.

   ```angular2html
    if __name__ == '__main__':
        # data segmentation
        start_pre_process(62, 62)
        """
        # extract features automatically
        start_extract_fea(is_normalization=True)
        """
   ```
2. Different groups of data might need different algorithms to segment. 
    ```angular2html
        def start(self):
            self.calc_real_fs()
            self.update_win_size()
            # self.get_index4(dependency="tidal", back_point=10)
            # self.get_index4(dependency="flow", back_point=0)
            # self.get_index4(dependency="pressure", back_point=3)
            self.get_index()
            self.save_data()
    ```
    The subfunction above belongs to class SegData. The default method is `self.get_index()` that can directly split the
    raw data set into breath-by-breath which may contain some improper segmentation.
    The code `self.get_index4()` is the entrance of another segmenting algorithm. It is up to the argument of dependency
    that can render specific method of segmentation.
3. The resulting file will be appeared in the directory ./original_data3/interested_data, ./original_data3/seg_data and 
    ./original _data3/pics/ after running the function start_pre_process().
  
### Data Annotation
1.  Visual inspection is the main method to annotate for breath cycle. The members of this work would not like share the
    source code we have developed for visually labelling which is independent upon other works,such as data segmentation,
    feature extraction. 

### Feature extraction
1. The same file mentioned above is able to extract features as follows.
    ```angular2html
    if __name__ == '__main__':
        """
        # data segmentation
        start_pre_process(62, 62)
        """
        # extract features automatically
        start_extract_fea(is_normalization=True, fea_name="approximate_en", n=0, m=3, r_coe=1)
    ```
    There are many arguments that is extremely of importance to extract non-linear features. It is necessary for the users
    to have to see the details of the four algorithms.
    ```angular2html
    is_normalization: normalize the raw data before extracting features if the value is True.
    fea_name: choose to extract which features among Fuzzy Entropy, Approximate Entropy, Sample Entropy, Poincare Plot.
    n: used in the fuzzy entropy, and its default value is 0.
    m: embeded dimension and its default value is 3. 
    r_coe: scaled-coefficient of stand deviation. 
    ```
2. The features extracted from the raw data were saved in the directory ./features/ after the procedure above.

*Pay attention to the file .gitkeep listed in directories ./raw_data/original_data3/, which is so useless for the work
above that the user ought to delete it.*

### Model training
1. Please turn on the file ./models/models.py and concentrate your eyes on the entrance of the main function. There are 
    also two parts in it. The first part is related to the traditional machine learning methods including Random Forest,
   Support Vector Machine, Logistic Regression, XGBoost and Voting, combined with four types of feature. The another refers
   to typical deep learning methods -- Convolutional Neural Network and its input is raw data stored in ./features/raw/.
   
   ```angular2html
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
        np.savetxt("final_rs.csv", final_rs, delimiter=",")
        """
        my_model = MyModel(model_name="cnn2", fea_name="raw", groups=[1, 2, 3, 4, 5, 6])
        my_model.start()
    """
   ```
2.  The argument of groups refers to SIX CASES of combinations between ventilation mode and preconfigured conditions of 
    lung simulator. The file of ./dataset/description/overview.xlsx detailed the content of each group in sheet2.
    
### Result saving
1. All the results were automatically stored in ./results/ after model training and evaluating.




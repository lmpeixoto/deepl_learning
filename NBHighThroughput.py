#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Imports
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif


def OS_HR(row):
    if row['HighRisk'] == 'HR' and row['OS_bin'] == 1:
        return 1
    elif row['HighRisk'] == 'HR' and row['OS_bin'] == 0:
        return 0


def EFS_HR(row):
    if row['HighRisk'] == 'HR' and row['EFS_bin'] == 1:
        return 1
    elif row['HighRisk'] == 'HR' and row['EFS_bin'] == 0:
        return 0


class HighThroughput:
    def __init__(self, ht_file, cd_file):
        """
        -ht_file: path of High Throughput expression file (str)
        -cd_file: path of Clinical Data file (str)

        """
        self.ht_file = ht_file
        self.cd_file = cd_file
        self.exprs = None
        self.clinical_data = None
        self.n_features = None
        self.n_samples = None
        self.features = None
        self.X = None
        self.y = None
        self.endpoint_name = None

    def read_exprs_data(self):
        """
        Reads high throughput expression data file.

        """
        print("Reading High Throughput gene expression file...")
        exprs = pd.read_table(self.ht_file, header=0, sep='\t')
        exprs_1 = exprs.filter(regex=("SEQC_")).transpose()
        exprs_1.columns = exprs['#Gene']
        self.exprs = exprs_1
        print("Expression data successfully load.")

    def read_clinical_data(self):
        """
        Reads clinical data file.

        """
        print("Reading clinical data file...")
        clinical_data = pd.read_csv(self.cd_file, sep=';', encoding="utf-8-sig")
        self.clinical_data = clinical_data
        print("Clinical data successfully load.")

    def HR_features(self):
        # self.clinical_data['OS_bin'] = np.asarray(self.clinical_data['OS_bin'], dtype="|S6")
        # self.clinical_data['EFS_bin'] = np.asarray(self.clinical_data['EFS_bin'], dtype="|S6")
        self.clinical_data['OS_HR'] = self.clinical_data.apply(lambda row: OS_HR(row), axis=1)
        self.clinical_data['EFS_HR'] = self.clinical_data.apply(lambda row: EFS_HR(row), axis=1)

    def set_feature_number(self):
        """
        Sets the number of features equal to the number of columns of expression matrix.

        """
        self.n_features = self.exprs.shape[1]

    def set_list_features(self):
        """
        Saves a list with the features of expression matrix.

        """
        self.features = list(self.exprs)

    def set_sample_number(self):
        """
        Sets the number of samples equal to the number of rows of expression matrix.

        """
        self.n_samples = self.exprs.shape[0]

    def load_data(self):
        """
        Runs previous functions to read all the data.

        """
        self.read_exprs_data()
        self.read_clinical_data()
        self.set_feature_number()
        self.set_list_features()
        self.set_sample_number()
        self.exprs.index = self.clinical_data['NB ID']
        self.HR_features()

    def get_feature_number(self):
        return self.n_features

    def get_list_features(self):
        return self.features

    def get_sample_number(self):
        return self.n_samples

    def variance_filter(self, variance):
        """
        Filter the columns data by variance threshold.

        """
        print("Filtering by variance of ", variance)
        before = self.get_feature_number()
        self.exprs = self.exprs.loc[:, (self.exprs.var() > int(variance))]
        self.set_feature_number()
        self.features = list(self.exprs)
        after = self.get_feature_number()
        print("Before: ", before)
        print("After: ", after)
        print("N feature filtered: ", before - after)

    def wilcoxon_filter(self, n_features):
        """
        Not finished
        - writes a wilcoxon sum rank by column and filter the top n_features pvalues"
        print("Filtering by Wilcoxon sum rank test. Number of features: ", n_features)

        """
        before = self.get_feature_number()
        pvalues = []
        for column in self.exprs:
            pvalues.append(stats.wilcoxon(self.exprs[column])[1])
        # retain the top 1000 p-values
        pvalues_copy = pvalues[:]
        pvalues_copy.sort()
        max_pvalue = pvalues_copy[-1000]
        mask = [i for i,v in enumerate(pvalues) if v >= max_pvalue]
        filtered = self.exprs.ix[mask]
        return filteredS

    def mse_filter(self, exprs, num_mad_genes):
        """
        Determine most variably expressed genes and subset

        """
        print('Determining most variably expressed genes and subsetting')
        mad_genes = exprs.mad(axis=0).sort_values(ascending=False)
        top_mad_genes = mad_genes.iloc[0:num_mad_genes, ].index
        subset_df = exprs.loc[:, top_mad_genes]
        return subset_df

    def filter_genes(self, exprs, y, number_genes):
        """
        Filter top number_genes using sklearn SelectKBest with filter f_classif

        """
        print('Filtering top ' + str(number_genes) + ' genes.')
        filter = SelectKBest(score_func=f_classif, k=number_genes)
        rnaseq_filtered = filter.fit(exprs, y).transform(exprs)
        mask = filter.get_support()
        new_features = exprs.columns[mask]
        rnaseq_filtered_df = pd.DataFrame(rnaseq_filtered, columns=new_features, index=exprs.index)
        return rnaseq_filtered_df

    def normalize_zero_one(self, exprs):
        """
        Scale expression data using zero-one normalization

        """
        print('Zero one data normalization.')
        rnaseq_scaled_zeroone_df = MinMaxScaler().fit_transform(exprs)
        rnaseq_scaled_zeroone_df = pd.DataFrame(rnaseq_scaled_zeroone_df,
                                                columns=exprs.columns,
                                                index=exprs.index)
        return rnaseq_scaled_zeroone_df

    def normalize_data(self, exprs):
        """
        Scale expression data using StandardScaler normalization

        """
        rnaseq_scaled_df = StandardScaler().fit_transform(exprs)
        rnaseq_scaled_df = pd.DataFrame(rnaseq_scaled_df,
                                                columns=exprs.columns,
                                                index=exprs.index)
        return rnaseq_scaled_df

    def save_matrices_train_test(self, X_train, X_test, y_train, y_test, root, file_name):
        if not os.path.exists(root):
            os.makedirs(root)
        i = 0
        X_train_name = os.path.join(root, 'X_train' + file_name + '.csv')
        X_test_name = os.path.join(root, 'X_test' + file_name + '.csv')
        y_train_name = os.path.join(root, 'y_train' + file_name + '.csv')
        y_test_name = os.path.join(root, 'y_test' + file_name + '.csv')
        np.savetxt(X_train_name, X_train)
        np.savetxt(X_test_name, X_test)
        np.savetxt(y_train_name, y_train)
        np.savetxt(y_test_name, y_test)

    def endpoint_Sex(self, split=0, features=5000, test_size=0.3):
        """

        :param  split = 0 - returns X,y not splited for k-fold cross-validation pipeline
                split = 1 - returns X_train, X_test, y_train, y_test with original split
                split = 2 - returns X_train, X_test, y_train, y_test and saves matrices

        :param features: number of final features (genes)

        :param test_size: test split factor

        :return: X, y or X_train, X_test, y_train, y_test matrices

        """
        self.endpoint_name = "endpoint_Sex"
        print("Clinical endpoint: Sex Imputed")
        X = self.exprs
        clinical_data = self.clinical_data
        X.index = clinical_data.index
        y = clinical_data['Sex_Imputed']
        y[y == 'M'] = 0
        y[y == 'F'] = 1
        X = self.normalize_data(X)
        X = self.mse_filter(X, features)
        if split == 0:
            print("Total: ", y.count())
            print("M: ", y.value_counts()[0])
            print("F: ", y.value_counts()[1])
            y = y.as_matrix().astype(int)
            X = X.as_matrix().astype(np.float)
            return X, y
        elif split == 1:
            X_train = X[clinical_data["Training/Validation set"] == 'Training'].as_matrix().astype(np.float)
            X_test = X[clinical_data["Training/Validation set"] == 'Validation'].as_matrix().astype(np.float)
            y_train = y.loc[clinical_data["Training/Validation set"] == 'Training']
            y_test = y.loc[clinical_data["Training/Validation set"] == 'Validation']
            print("Total: ", (y_train.count() + y_test.count()))
            print("M: ", (len(y_train[y_train == 1]) + len(y_test[y_test == 1])))
            print("F: ", (len(y_train[y_train == 0]) + len(y_test[y_test == 0])))
            y_train = y_train.as_matrix().astype(int)
            y_test = y_test.as_matrix().astype(int)
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
            # Save *.csv files with stratified train/test subsets
            root = 'Endpoints_Xy/endpoint_Sex'
            file_name = '_NB_' + str(features)
            self.save_matrices_train_test(X_train, X_test, y_train, y_test, root, file_name)
            return X_train, X_test, y_train, y_test

    def endpoint_EFSAll(self, split=0, features=5000, test_size=0.3):
        self.endpoint_name = "endpoint_EFSAll"
        print("Clinical endpoint: EFS All")
        X = self.exprs
        clinical_data = self.clinical_data
        y = self.clinical_data['EFS_bin']
        X = self.normalize_data(X)
        X = self.mse_filter(X, features)
        if split == 0:
            print("Total: ", y.count())
            print("1: ", y.value_counts()[0])
            print("0: ", y.value_counts()[1])
            y = y.as_matrix().astype(int)
            X = X.as_matrix().astype(np.float)
            return X, y
        elif split == 1:
            X_train = X[clinical_data["Training/Validation set"] == 'Training'].as_matrix().astype(np.float)
            X_test = X[clinical_data["Training/Validation set"] == 'Validation'].as_matrix().astype(np.float)
            y_train = y.loc[clinical_data["Training/Validation set"] == 'Training']
            y_test = y.loc[clinical_data["Training/Validation set"] == 'Validation']
            print("Total: ", (y_train.count() + y_test.count()))
            print("1: ", (len(y_train[y_train == 1]) + len(y_test[y_test == 1])))
            print("0: ", (len(y_train[y_train == 0]) + len(y_test[y_test == 0])))
            y_train = y_train.as_matrix().astype(int)
            y_test = y_test.as_matrix().astype(int)
            self.endpoint_name = "endpoint_EFSAll"
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
            # Save *.csv files with stratified train/test subsets
            root = 'Endpoints_Xy/endpoint_EFSAll'
            file_name = '_NB_' + str(features)
            self.save_matrices_train_test(X_train, X_test, y_train, y_test, root, file_name)
            return X_train, X_test, y_train, y_test

    def endpoint_ClassLabel(self, split=0, features=5000, test_size=0.3):
        self.endpoint_name = "endpoint_ClassLabel"
        print("Clinical endpoint: Class Label")
        X = self.exprs
        clinical_data = self.clinical_data
        X.index = clinical_data.index
        X = X[~clinical_data['Class label'].isnull()]
        y = clinical_data[~clinical_data['Class label'].isnull()]
        y = y['Class label']
        y[y == 'favorable'] = 0
        y[y == 'unfavorable'] = 1
        X = self.normalize_data(X)
        X = self.mse_filter(X, features)
        if split == 0:
            print("Total: ", y.count())
            print("Favorable: ", y.value_counts()[0])
            print("Unfavorable: ", y.value_counts()[1])
            y = y.as_matrix().astype(int)
            X = X.as_matrix().astype(np.float)
            return X, y
        elif split == 1:
            X_train = X[clinical_data["Training/Validation set"] == 'Training'].as_matrix().astype(np.float)
            X_test = X[clinical_data["Training/Validation set"] == 'Validation'].as_matrix().astype(np.float)
            y_train = y.loc[clinical_data["Training/Validation set"] == 'Training']
            y_test = y.loc[clinical_data["Training/Validation set"] == 'Validation']
            print("Total: ", (y_train.count() + y_test.count()))
            print("1: ", (len(y_train[y_train == 1]) + len(y_test[y_test == 1])))
            print("0: ", (len(y_train[y_train == 0]) + len(y_test[y_test == 0])))
            y_train = y_train.as_matrix().astype(int)
            y_test = y_test.as_matrix().astype(int)
            self.endpoint_name = "endpoint_ClassLabel"
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
            # Save *.csv files with stratified train/test subsets
            root = 'Endpoints_Xy/endpoint_ClassLabel'
            file_name = '_NB_' + str(features)
            self.save_matrices_train_test(X_train, X_test, y_train, y_test, root, file_name)
            return X_train, X_test, y_train, y_test

    def endpoint_OSAll(self, split=0, features=5000, test_size=0.3):
        self.endpoint_name = "endpoint_OSAll"
        print("Clinical endpoint: OS All")
        X = self.exprs
        clinical_data = self.clinical_data
        X.index = clinical_data.index
        y = clinical_data['OS_bin']
        X = self.normalize_data(X)
        X = self.mse_filter(X, features)
        if split == 0:
            print("Total: ", y.count())
            print("1: ", y.value_counts()[0])
            print("0: ", y.value_counts()[1])
            y = y.as_matrix().astype(int)
            X = X.as_matrix().astype(np.float)
            return X, y
        elif split == 1:
            X_train = X[clinical_data["Training/Validation set"] == 'Training'].as_matrix().astype(np.float)
            X_test = X[clinical_data["Training/Validation set"] == 'Validation'].as_matrix().astype(np.float)
            y_train = y.loc[clinical_data["Training/Validation set"] == 'Training']
            y_test = y.loc[clinical_data["Training/Validation set"] == 'Validation']
            print("Total: ", (y_train.count() + y_test.count()))
            print("1: ", (len(y_train[y_train == 1]) + len(y_test[y_test == 1])))
            print("0: ", (len(y_train[y_train == 0]) + len(y_test[y_test == 0])))
            y_train = y_train.as_matrix().astype(int)
            y_test = y_test.as_matrix().astype(int)
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
            # Save *.csv files with stratified train/test subsets
            root = 'Endpoints_Xy/endpoint_OSAll'
            file_name = '_NB_' + str(features)
            self.save_matrices_train_test(X_train, X_test, y_train, y_test, root, file_name)
            return X_train, X_test, y_train, y_test

    def endpoint_OSHR(self, split=0, features=5000, test_size=0.3):
        self.endpoint_name = "endpoint_OSHR"
        print("Clinical endpoint: OS HR")
        X = self.exprs
        clinical_data = self.clinical_data
        X.index = clinical_data.index
        X = X[clinical_data['HighRisk'] == 'HR']
        y = clinical_data[clinical_data['HighRisk'] == 'HR']
        y = y['OS_bin']
        X = self.normalize_data(X)
        X = self.mse_filter(X, features)
        if split == 0:
            print("Total: ", y.count())
            print("1: ", y.value_counts()[0])
            print("0: ", y.value_counts()[1])
            y = y.as_matrix().astype(int)
            X = X.as_matrix().astype(np.float)
            return X, y
        elif split == 1:
            X_train = X[clinical_data["Training/Validation set"] == 'Training'].as_matrix().astype(np.float)
            X_test = X[clinical_data["Training/Validation set"] == 'Validation'].as_matrix().astype(np.float)
            y_train = y.loc[clinical_data["Training/Validation set"] == 'Training']
            y_test = y.loc[clinical_data["Training/Validation set"] == 'Validation']
            print("Total: ", (y_train.count() + y_test.count()))
            print("1: ", (len(y_train[y_train == 1]) + len(y_test[y_test == 1])))
            print("0: ", (len(y_train[y_train == 0]) + len(y_test[y_test == 0])))
            y_train = y_train.as_matrix().astype(int)
            y_test = y_test.as_matrix().astype(int)
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
            # Save *.csv files with stratified train/test subsets
            root = 'Endpoints_Xy/endpoint_OSHR'
            file_name = '_NB_' + str(features)
            self.save_matrices_train_test(X_train, X_test, y_train, y_test, root, file_name)
            return X_train, X_test, y_train, y_test

    def endpoint_EFSHR(self, split=0, features=5000, test_size=0.3):
        self.endpoint_name = "endpoint_EFSHR"
        print("Clinical endpoint: EFS HR")
        X = self.exprs
        clinical_data = self.clinical_data
        X.index = clinical_data.index
        X = X[clinical_data['HighRisk'] == 'HR']
        y = clinical_data[clinical_data['HighRisk'] == 'HR']
        y = y['EFS_bin']
        X = self.normalize_data(X)
        X = self.mse_filter(X, features)
        if split == 0:
            print("Total: ", y.count())
            print("1: ", y.value_counts()[0])
            print("0: ", y.value_counts()[1])
            y = y.as_matrix().astype(int)
            X = X.as_matrix().astype(np.float)
            return X, y
        elif split == 1:
            X_train = X[clinical_data["Training/Validation set"] == 'Training'].as_matrix().astype(np.float)
            X_test = X[clinical_data["Training/Validation set"] == 'Validation'].as_matrix().astype(np.float)
            y_train = y.loc[clinical_data["Training/Validation set"] == 'Training']
            y_test = y.loc[clinical_data["Training/Validation set"] == 'Validation']
            print("Total: ", (y_train.count() + y_test.count()))
            print("1: ", (len(y_train[y_train == 1]) + len(y_test[y_test == 1])))
            print("0: ", (len(y_train[y_train == 0]) + len(y_test[y_test == 0])))
            y_train = y_train.as_matrix().astype(int)
            y_test = y_test.as_matrix().astype(int)
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
            # Save *.csv files with stratified train/test subsets
            root = 'Endpoints_Xy/endpoint_EFSHR'
            file_name = '_NB_' + str(features)
            self.save_matrices_train_test(X_train, X_test, y_train, y_test, root, file_name)
            return X_train, X_test, y_train, y_test

    def multi_task(self, split=0, features=5000, test_size=0.3):
        print("Multi task network")
        self.exprs.index = self.clinical_data.index
        y = self.clinical_data.ix[:, ('Sex_Imputed', 'Class label', 'EFS_bin', 'OS_bin', 'EFS_HR', 'OS_HR')]
        y = pd.get_dummies(y)
        y['Sex_Imputed'] = y['Sex_Imputed_F']
        y['Class label'] = y['Class label_unfavorable']
        y.drop(['Sex_Imputed_M', 'Sex_Imputed_F', 'Class label_unfavorable', 'Class label_favorable'], inplace=True,
               axis=1)
        y.index = self.exprs.index
        y = y.dropna()
        X = self.exprs.ix[y.index].dropna(axis=1)
        X = self.normalize_data(X)
        X = self.mse_filter(X, features)
        # X = self.filter_genes(X, y, features)
        # imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
        # imp.fit(y)
        # y = imp.transform(y)
        y = y.as_matrix().astype(int)
        X = X.as_matrix().astype(np.float)
        self.endpoint_name = "multi-task"
        return X, y

import logging
import sys

import pandas as pd
import numpy as np
import csv as csv
import featuretools as ft

from pprint import pprint
from datacleaner import autoclean
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import shuffle

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class DataSet(object):

    def __init__(self, train_file, test_file, predict_var):
        self.train_file = train_file
        self.test_file = test_file
        self.training_data = pd.read_csv(train_file)
        self.testing_data = pd.read_csv(test_file)
        self.predict_var = predict_var

    def set_data(self, new_training_data, new_testing_data):
        """
        Setter for replacing both training/testing datasets
        """
        self.training_data = new_training_data
        self.testing_data = new_testing_data

    def get_data(self):
        """
        Get the training_data and testing_data
        """
        return self.training_data, self.testing_data

    def add_column(self, cols):
        """
        Add columns to both training_data and testing_data
        """
        for col in cols:
            self.training_data.join(col)
            self.testing_data.join(col)
        return self.training_data, self.testing_data

    def drop_columns(self, cols):
        """
        Drop columns from both training_data and testing_data 
        """
        self.training_data.drop(cols, axis=1)
        self.testing_data.drop(cols, axis=1)
        return self.training_data, self.testing_data

    def check_training_unique(self, field):
        """
        Check if a all fields are unique in training_data.
        """
        values_unique = False
        if self.training_data[field].nunique() == self.training_data.shape[0]:
            logging.info('All {} are unique in the training set.'.format(field))
            values_unique = True
        else:
            logging.info('NOT all {} are unique in the training set.'.format(field))
        return values_unique

    def check_training_and_testing_are_unique(self, field):
        """
        Check if all fields are unique between training and testing
        """
        values_unique = False
        if len(np.intersect1d(self.training_data[field].values, self.testing_data[field].values)) == 0:
            logging.info('The training and testing datasets have none of the same {} values.'.format(field))
            values_unique = True
        else:
            logging.info('The training and testing datasets have none of the same {} values.'.format(field))
        return values_unique

    def check_for_nan(self):
        """
        Check if either the training or testing datasets have a NaN value
        """
        NaN_found = False
        if self.training_data.count().min() == self.training_data.shape[0] and self.testing_data.count().min() == self.testing_data.shape[0]:
            logging.info('There are no NaN values in the datasets.')
        else:
            NaN_found = True
            logging.info('A NaN value was found!')
            nas = pd.concat([self.training_data.isnull().sum(), self.testing_data.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset'])
            logging.info('Nan in the data sets')
            logging.info(nas[nas.sum(axis=1) > 0])
        return NaN_found

    def get_training_data_info(self):
        """
        Returns information about the training data
        """
        dtype_df = self.training_data.dtypes.reset_index()
        dtype_df.columns = ["Count", "Column Type"]
        dtype_df.groupby("Column Type").aggregate('count').reset_index()
        return dtype_df

    def clean_data(self, replace_data=False):
        """
        Attempt to clean the training_data and testing_data using datacleaner.autoclean
        """
        clean_training, clean_testing = autoclean(self.training_data), autoclean(self.testing_data)

        if replace_data:
            self.set_data(clean_training, clean_testing)

        return clean_training, clean_testing

    def perform_feature_engineering(self):
        es = ft.EntitySet(id='Titanic')
        es.entity_from_dataframe(entity_id='training', dataframe=self.training_data, index='PassengerId')
        print(es)
        feature_matrix, feature_names = ft.dfs(entityset=es, target_entity = 'training', max_depth = 2, verbose = 1, n_jobs = 3)
        print(feature_names)

    def get_training_data(self):
        """
        Manipulate the data for training
        """
        X_train = self.training_data.drop(self.predict_var, axis=1)
        Y_train = self.training_data[self.predict_var]

        return X_train, Y_train


class Model(object):

    def __init__(self, X_train, Y_train, X_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

    def fit(self):
        return self.estimator.fit(self.X_train, self.Y_train)

    def predict(self):
        return self.estimator.predict(self.X_test)

    def score(self):
        return self.estimator.score(self.X_train, self.Y_train)

    def cross_validation_score(self, cv=10):
        return cross_val_score(self.estimator, self.X_train, self.Y_train, cv=cv).mean()

    def run_model(self):
        self.fit()
        self.predict()
        result_train = self.score()
        result_val = self.cross_validation_score()
        print('training score = %s , while validation score = %s' %(result_train , result_val))
        return result_train, result_val


class LogisticRegressionModel(Model):

    def __init__(self, X_train, Y_train, X_test):
        super(LogisticRegressionModel, self).__init__(X_train, Y_train, X_test)
        self.estimator = LogisticRegression()


class SVMModel(Model):

    def __init__(self, X_train, Y_train, X_test, C=0.1, gamma=0.1):
        super(SVMModel, self).__init__(X_train, Y_train, X_test)
        self.estimator = SVC(C=C, gamma=gamma)


class RandomForestModel(Model):

    def __init__(self, X_train, Y_train, X_test, n_estimators=1000, criterion='gini', min_samples_split=10, min_samples_leaf=1, max_features='auto', oob_score=True, random_state=1, n_jobs=1):
        super(RandomForestModel, self).__init__(X_train, Y_train, X_test)
        self.estimator = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, oob_score=oob_score, random_state=random_state, n_jobs=n_jobs)

class Autopilot(object):

    def __init__(self, data):
        if not isinstance(data, DataSet):
            raise Exception
        self.data = data
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.results = []
        self.best_model = None

    def prepare_data(self):
        self.data.clean_data(replace_data=True)
        self.X_train, self.Y_train = self.data.get_training_data()
        self.X_test = self.data.testing_data

    def set_data(self, X_train, Y_train, X_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test

    def run_autopilot(self, skip_preparation=False):
        print("****************************************************")
        print("Autopilot Started")

        if not skip_preparation:
            self.prepare_data()

        # Logistic Regression
        print("Running Logistic Regression")
        logreg = LogisticRegressionModel(self.X_train, self.Y_train, self.X_test)
        training_score, validation_score = logreg.run_model()
        self.results.append({
            'model': 'Logistic Regression',
            'training score': training_score,
            'validation score': validation_score,
        })

        # SVM
        print("Running SVM")
        svm = SVMModel(self.X_train, self.Y_train, self.X_test)
        training_score, validation_score = svm.run_model()
        self.results.append({
            'model': 'SVM',
            'training score': training_score,
            'validation score': validation_score
        })

        # Random Forests
        print("Running Random Forests")
        n_estimators=100
        random_forest = RandomForestModel(self.X_train, self.Y_train, self.X_test, n_estimators=n_estimators)
        training_score, validation_score = random_forest.run_model()
        self.results.append({
            'model': 'Random Forests ({} Estimators)'.format(n_estimators),
            'training score': training_score,
            'validation score': validation_score
        })

        list.sort(self.results, key=lambda x: x['validation score'], reverse=True)
        self.best_model = self.results[0]

        print("Autopilot Finished")
        print("******LEADERBOARD******")
        pprint(self.results)
        print("******BEST MODEL******")
        pprint(self.best_model)
        print("****************************************************")

        return self.results, self.best_model

###################################################################################################

titanic = DataSet('train.csv', 'test.csv', 'Survived')
titanic.clean_data(replace_data=True)
titanic.drop_columns(['Embarked'])
titanic_autopilot = Autopilot(titanic)
X_train, Y_train = titanic.get_training_data()
X_test = titanic.testing_data
titanic_autopilot.set_data(X_train, Y_train, X_test)
results, best_model = titanic_autopilot.run_autopilot()


# titanic.clean_data(replace_data=True)
# X_train, Y_train = titanic.get_training_data()
# X_test = titanic.testing_data



# Y_pred = random_forest.predict()

###################################################################################################

# submission = pd.DataFrame({
#         "PassengerId": X_test["PassengerId"],
#         "Survived": Y_pred
#     })
# submission.to_csv('titanic.csv', index=False)
# print('Exported')

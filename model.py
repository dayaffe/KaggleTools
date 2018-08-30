import logging
import sys

import pandas as pd
import numpy as np
import csv as csv
# import featuretools as ft

from pprint import pprint
from datacleaner import autoclean
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
# from xgboost import XGBClassifier

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

    # def perform_feature_engineering(self):
    #     es = ft.EntitySet(id='Titanic')
    #     es.entity_from_dataframe(entity_id='training', dataframe=self.training_data, index='PassengerId')
    #     print(es)
    #     feature_matrix, feature_names = ft.dfs(entityset=es, target_entity = 'training', max_depth = 2, verbose = 1, n_jobs = 3)
    #     print(feature_names)

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
        self.Y_pred = None
        self.estimator = None
        self.model_name = None
        self.training_score = None
        self.validation_score = None
        self.model_info = None

    def fit(self):
        return self.estimator.fit(self.X_train, self.Y_train)

    def predict(self):
        self.Y_pred = self.estimator.predict(self.X_test)
        return self.Y_pred

    def score(self):
        return self.estimator.score(self.X_train, self.Y_train)

    def cross_validation_score(self, cv=10):
        return cross_val_score(self.estimator, self.X_train, self.Y_train, cv=cv).mean()

    def run_model(self):
        self.fit()
        self.predict()
        self.training_score = self.score()
        self.validation_score = self.cross_validation_score()
        print('training score = %s , while validation score = %s' %(self.training_score , self.validation_score))
        return self.training_score, self.validation_score

    def get_model(self):
        """
        Return everything a user would want about the model
        """
        self.model_info = {
            'model': self.model_name,
            'estimator': self.estimator,
            'X_train': self.X_train,
            'Y_train': self.Y_train,
            'X_test': self.X_test,
            'Y_pred': self.Y_pred,
            'training_score': self.training_score,
            'validation_score': self.validation_score
        }
        return self.model_info

    def print_model(self):
        """
        Print the model info for the leaderboard
        """
        print('Model: {}, Training Score: {}, Validation Score: {}'.format(self.model_name, self.training_score, self.validation_score))

    @staticmethod
    def submit(estimator, id_col, predict_col, filename, index=False):
        submission = pd.DataFrame({
                id_col: X_test[id_col],
                predict_col: estimator.Y_pred
            })
        submission.to_csv(filename, index=index)
        print('Exported')


class LogisticRegressionModel(Model):

    def __init__(self, X_train, Y_train, X_test):
        super(LogisticRegressionModel, self).__init__(X_train, Y_train, X_test)
        self.estimator = LogisticRegression()
        self.model_name = 'Logistic Regression'


class SVMModel(Model):

    def __init__(self, X_train, Y_train, X_test, C=30, gamma=0.01):
        super(SVMModel, self).__init__(X_train, Y_train, X_test)
        self.estimator = SVC(C=C, gamma=gamma)
        self.model_name = 'SVM'


class NaiveBayesModel(Model):

    def __init__(self, X_train, Y_train, X_test):
        super(NaiveBayesModel, self).__init__(X_train, Y_train, X_test)
        self.estimator = GaussianNB()
        self.model_name = 'Naive Bayes (GaussianNB)'


# class XGBoostModel(Model):

#     def __init__(self, X_train, Y_train, X_test, base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
#        max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
#        n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
#        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#        silent=True, subsample=1):
#        super(XGBoostModel, self).__init__(X_train, Y_train, X_test)
#        self.estimator = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
#        max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
#        n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
#        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#        silent=True, subsample=1)
#        self.model_name = 'XGBoost Classifier'


class RandomForestModel(Model):

    def __init__(self, X_train, Y_train, X_test, n_estimators=1000, criterion='gini', min_samples_split=10, min_samples_leaf=1, max_features='auto', oob_score=True, random_state=1, n_jobs=1):
        super(RandomForestModel, self).__init__(X_train, Y_train, X_test)
        self.estimator = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, oob_score=oob_score, random_state=random_state, n_jobs=n_jobs)
        self.model_name = 'Random Forest ({} estimators)'.format(n_estimators)

class BlenderModel(Model):

    def __init__(self, models):
        for model in models:
            if not isinstance(model, Model):
                raise Exception
            if len(models) % 2 == 0:
                raise Exception
        self.models = models
        self.model_count = len(models)
        self.estimator = self.models[0] # we just need an estimator/X_train to use sklearn's score()
        self.X_train = self.estimator.X_train
        self.Y_pred = []
        self.Y_train = []
        self.model_name = 'Blender of {}'.format([model.model_name for model in self.models])

    def blend(self):
        """
        Blend using majority rules
        """
        votes = []
        training_predictions = []
        testing_predictions = []

        train_row_count = len(self.models[0].Y_train)
        pred_row_count = len(self.models[0].Y_pred)

        for row in range(train_row_count):
            votes = [model.Y_train[row] for model in self.models]
            self.Y_train.append(max(set(votes), key=votes.count))

        for row in range(pred_row_count):
            votes = [model.Y_pred[row] for model in self.models]
            self.Y_pred.append(max(set(votes), key=votes.count))

        self.Y_train = np.array(self.Y_train, dtype=np.int)
        self.Y_pred = np.array(self.Y_pred, dtype=np.int)

    def run_model(self):
        self.blend()
        self.training_score = self.score()
        self.validation_score = None
        print('training score = %s , while validation score = %s' %(self.training_score , self.validation_score))
        return self.training_score, self.validation_score

    def score(self):
        return self.estimator.score()

    def get_model(self):
        """
        Return everything a user would want about the model
        """
        self.model_info = {
            'model': self.model_name,
            'model_count': self.model_count,
            'Y_train': self.Y_train,
            'Y_pred': self.Y_pred,
            'training_score': self.training_score,
            'validation_score': self.validation_score
        }
        return self.model_info


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
        print("Autopilot Starting...")

        if not skip_preparation:
            self.prepare_data()

        # Logistic Regression
        print("Running Logistic Regression")
        logreg = LogisticRegressionModel(self.X_train, self.Y_train, self.X_test)
        training_score, validation_score = logreg.run_model()
        print(logreg.Y_pred)
        self.results.append(logreg)

        # SVM
        print("Running SVM")
        svm = SVMModel(self.X_train, self.Y_train, self.X_test)
        training_score, validation_score = svm.run_model()
        print(svm.Y_pred)
        self.results.append(svm)

        # Random Forests
        print("Running Random Forests")
        n_estimators=100
        random_forest = RandomForestModel(self.X_train, self.Y_train, self.X_test, n_estimators=n_estimators)
        training_score, validation_score = random_forest.run_model()
        print(random_forest.Y_pred)
        self.results.append(random_forest)

        # Naive Bayes
        print("Running Naive Bayes (GaussianNB)")
        naive_bayes = NaiveBayesModel(self.X_train, self.Y_train, self.X_test)
        training_score, validation_score = naive_bayes.run_model()
        print(naive_bayes.Y_pred)
        self.results.append(naive_bayes)

        # Blender
        print("Running Blender")
        blender_model = BlenderModel([logreg, random_forest, naive_bayes])
        blender_model.run_model()
        print(blender_model.Y_pred)
        self.results.append(blender_model)

        list.sort(self.results, key=lambda x: x.get_model()['validation_score'], reverse=True)
        self.best_model = self.results[0]

        print("Autopilot Finished")
        print("******LEADERBOARD******")
        for result in self.results:
            result.print_model()
        print("******BEST MODEL******")
        self.best_model.print_model()
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
Model.submit(results[4], "PassengerId", "Survived", "titanic.csv")


# titanic.clean_data(replace_data=True)
# X_train, Y_train = titanic.get_training_data()
# X_test = titanic.testing_data



# Y_pred = random_forest.predict()

###################################################################################################

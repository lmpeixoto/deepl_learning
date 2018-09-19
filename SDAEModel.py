#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# Imports

from NBMicroarrays import *
from external_functions import *
from numpy import *
from random import choice
import matplotlib
import time
import DNNModel
matplotlib.use('Agg')
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Input
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold, KFold
from NBHighThroughput import *
from external_functions import *
from numpy import *
from random import choice
import matplotlib
matplotlib.use('Agg')
import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, \
    log_loss




class SDAEModel():
    def __init__(self, **kwargs):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.splitted = None
        self.model = None
        self.verbose = 0
        if validate_matrices(kwargs):
            if len(kwargs.keys()) <= 3:
                self.X = kwargs['X']
                self.y = kwargs['y']
                self.splitted = False
            elif len(kwargs.keys()) > 3:
                self.X_train = kwargs['X_train']
                self.X_test = kwargs['X_test']
                self.y_train = kwargs['y_train']
                self.y_test = kwargs['y_test']
                self.splitted = True
        if kwargs['cv']:
            assert type(kwargs['cv']) is int, 'cv value must be of type int.'
            assert kwargs['cv'] >= 3, 'cv value must be at least 3.'
            self.cv = kwargs['cv']
        if self.splitted == True:
            self.feature_number = self.X_train.shape[1]
        elif self.splitted == False:
            self.feature_number = self.X.shape[1]

        self.parametersAE = {

            'output_activation': 'sigmoid',
            'optimization': 'SGD',
            'learning_rate': 0.001,
            'units_in_input_layer': 5000,
            'units_in_hidden_layers': [2000, 500, 2000],
            'nb_epoch': 5,
            'batch_size': 60,
            'early_stopping': False,
            'patience': 30

        }

        self.parametersDNN = {

            'dropout': 0.2,
            'output_activation': 'sigmoid',
            'optimization': 'SGD',
            'learning_rate': 0.001,
            'units_in_input_layer': 500,
            'units_in_hidden_layers': [100, 50],
            'nb_epoch': 100,
            'batch_size': 60,
            'early_stopping': False,
            'patience': 60

        }

        self.filename = None

        self.parameters_batch_AE = {

            'output_activation': ['sigmoid'],
            'optimization': ['SGD', 'Adam', 'RMSprop'],
            'learning_rate': [0.001, 0.01],
            'batch_size': [60, 80, 100, 120],
            'nb_epoch': [800],
            'units_in_hidden_layers': [[4000, 1000, 4000], [4000, 2000, 100, 2000, 4000],
                                       [4000, 2000, 1000, 100, 50, 100, 1000, 2000, 4000], [2500, 750, 50, 750, 2500],
                                       [2500, 250, 50, 250, 2500], [1000, 100, 50, 100, 1000], [1000, 50, 1000]],
            'units_in_input_layer': [5000],
            'early_stopping': [True],
            'patience': [80]
        }

        self.parameters_batch = {

            'output_activation': ['sigmoid', 'tanh'],
            'optimization': ['SGD'],
            'learning_rate': [0.015, 0.010, 0.005, 0.001],
            'batch_size': [30, 60, 75, 80, 100, 120],
            'nb_epoch': [500],
            'units_in_hidden_layers': [[4000, 1000, 10], [4000, 2000, 1000, 10], [4000, 2000, 1000, 100, 10],
                                       [2500, 750, 10], [2500, 250, 10], [1000, 100, 10], [1000, 10]],
            'units_in_input_layer': [5000],
            'early_stopping': [True],
            'patience': [20]
        }

        self.model_selection_history = []

    def print_parameter_values_AE(self):
        print("Hyperparameters")
        for key in sorted(self.parametersAE):
            print(key + ": " + str(self.parametersAE[key]))

    def create_SDAE_model(self):
        print("Creating SDAE model")
        fundamental_parameters = ['output_activation', 'optimization', 'learning_rate', 'units_in_input_layer',
                                  'units_in_hidden_layers', 'nb_epoch', 'batch_size']
        for param in fundamental_parameters:
            if self.parametersAE[param] == None:
                print("Parameter not set: " + param)
                return
        self.print_parameter_values_AE()
        inputs = Input(shape=(self.feature_number,))
        encoded = None
        # Hidden layers
        encoded = Dense(self.parametersAE['units_in_input_layer'], activation='relu')(inputs)
        encoded = BatchNormalization()(encoded)
        if type(self.parametersAE['units_in_hidden_layers']) == list:
            for layer in self.parametersAE['units_in_hidden_layers']:
                if encoded:
                    encoded = Dense(layer, activation='relu')(encoded)
                    encoded = BatchNormalization()(encoded)
                else:
                    encoded = Dense(layer, activation='relu')
                    encoded = BatchNormalization()(encoded)
        else:
            layer = self.parametersAE['units_in_hidden_layers']
            if encoded:
                encoded = Dense(layer, activation='relu')(encoded)
                encoded = BatchNormalization()(encoded)
            else:
                encoded = Dense(layer, activation='relu')
                encoded = BatchNormalization()(encoded)

        encoded = Dense(self.parametersAE['units_in_input_layer'], activation='relu')(encoded)
        encoded = BatchNormalization()(encoded)
        # Output layer
        outputs = Dense(self.feature_number, activation='relu')(encoded)
        outputs = BatchNormalization()(outputs)
        autoencoder = Model(input=inputs, output=outputs)
        supervised = Model(input=inputs, output=encoded)
        # Optimization
        if self.parametersAE['optimization'] == 'SGD':
            optim = SGD()
            optim.lr.set_value(self.parametersAE['learning_rate'])
        elif self.parametersAE['optimization'] == 'RMSprop':
            optim = RMSprop()
            optim.lr.set_value(self.parametersAE['learning_rate'])
        elif self.parametersAE['optimization'] == 'Adam':
            optim = Adam()
        elif self.parametersAE['optimization'] == 'Adadelta':
            optim = Adadelta()
        # Compiling autoencoder
        autoencoder.compile(loss='mse', optimizer=optim)
        supervised.compile(loss='mse', optimizer=optim)
        if self.verbose == 1: str(autoencoder.summary())
        self.autoencoder = autoencoder
        self.encoder = supervised
        print("SDAE model sucessfully created")



    def fit_SDAE_model(self, X):
        print("Fitting SDAE model")
        start_time = time.time()
        if self.parametersAE['nb_epoch'] and self.parametersAE['batch_size']:
            if self.parametersAE['early_stopping']:
                early_stopping = EarlyStopping(monitor='val_loss', patience=self.parametersAE['patience'])
                self.history = self.autoencoder.fit(X, X, epochs=self.parametersAE['nb_epoch'],
                                                    batch_size=self.parametersAE['batch_size'],
                                                    verbose=self.verbose,
                                                    callbacks=[early_stopping])
            else:
                self.history = self.autoencoder.fit(X, X, epochs=self.parametersAE['nb_epoch'],
                                                    batch_size=self.parametersAE['batch_size'],
                                                    verbose=self.verbose)
        fit_time = time.time() - start_time
        print("SDAE model successfully fit in ", timer(fit_time))
        return fit_time


    def print_fit_results(self):
        print('val_loss: ', min(self.history.history['val_loss']))
        print('train_loss: ', min(self.history.history['loss']))
        print("train/val loss ratio: ", min(self.history.history['loss']) / min(self.history.history['val_loss']))

    def batch_parameter_shufller_AE(self):
        chosen_param = {}
        for key in self.parameters_batch_AE:
            chosen_param[key] = choice(self.parameters_batch_AE[key])
        return chosen_param

    def model_selection_AE(self, X, y, n_iter_search=2, n_folds=2):
        print("Selecting best SDAE model")
        seed = 7
        old_parameters = self.parametersAE.copy()
        self.model_selection_history = []
        for iteration in range(n_iter_search):
            mean_train_loss = []
            mean_val_loss = []
            mean_time_fit = []
            print("Iteration no. " + str(iteration + 1))
            new_parameters = self.batch_parameter_shufller_AE()
            temp_values = new_parameters.copy()
            for key in new_parameters:
                self.parametersAE[key] = new_parameters[key]
            i = 0
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for train, test in kf.split(X):
                print("Running Fold " + str(i + 1) + str("/") + str(n_folds))
                X_train = X[train,]
                X_test = X[test,]
                self.create_SDAE_model()
                time_fit = self.fit_SDAE_model(X_train, X_test)
                self.print_fit_results()
                train_loss = min(self.history.history['loss'])
                val_loss = min(self.history.history['val_loss'])
                mean_train_loss.append(train_loss)
                mean_val_loss.append(val_loss)
                mean_time_fit.append(time_fit)
                temp_values['train_loss_' + str(i + 1)] = train_loss
                temp_values['val_loss_' + str(i + 1)] = val_loss
                temp_values['time_fit_' + str(i + 1)] = time_fit
                i += 1
            temp_values['mean_train_loss'] = np.mean(mean_train_loss)
            temp_values['mean_val_loss'] = np.mean(mean_val_loss)
            temp_values['mean_time_fit'] = np.mean(mean_time_fit)
            self.model_selection_history.append(temp_values)
        self.parametersAE = old_parameters.copy()
        print("Best SDAE model successfully selected")

    def find_best_model_AE(self):
        if self.model_selection_history:
            best_model = None
            min_val_loss = 0
            for dic in self.model_selection_history:
                if dic['mean_val_loss'] < min_val_loss:
                    best_model = dic
                    min_val_loss = dic['mean_val_loss']
        if best_model == None:
            best_model = self.model_selection_history[0]
        print("Best model:")
        self.print_parameter_values()
        print("Loss: " + str(min_val_loss))
        return best_model

    def select_best_model_AE(self):
        best_model = self.find_best_model_AE()
        for key in self.parametersAE:
            self.parametersAE[key] = best_model[key]

    def write_model_selection_results_AE(self):
        d = self.model_selection_history
        sequence = ['mean_val_loss', 'mean_train_loss', 'mean_time_fit']
        sequence_tv_scores = [key for key in d[0] if key.startswith(("train_", "val_", "time_"))]
        sequence_tv_scores.sort()
        sequence_parameters = [x for x in self.parametersAE]
        sequence_parameters.sort()
        sequence.extend(sequence_tv_scores + sequence_parameters)
        df = pd.DataFrame(d, columns=sequence).sort_values(['mean_val_loss'], ascending=[True])
        root = 'SDAE_Model_Selection_Results' + '/' + self.filename
        if not os.path.exists(root):
            os.makedirs(root)
        i = 0
        while os.path.exists(root + '/' + self.filename + '_model_selection_table_' + str(i) + '.csv'):
            i += 1
        file_name = os.path.join(root, self.filename + '_model_selection_table_' + str(i) + '.csv')
        print('Writing csv file with path: ', file_name)
        df.to_csv(file_name, sep='\t')
        self.model_selection_results = df

    def write_report_AE(self, metric_scores):
        root = 'SDAE_Model_Selection_Results' + '/' + self.filename
        if not os.path.exists(root):
            os.makedirs(root)
        i = 0
        while os.path.exists(root + '/' + self.filename + '_report_' + str(i) + '.txt'):
            i += 1
        file_name = os.path.join(root, self.filename + '_report_' + str(i) + '.txt')
        print("Writing report file with path: " + file_name)
        out = open(file_name, 'w')
        out.write('SDAE Hyperparameters')
        out.write('\n')
        out.write('=' * 25)
        out.write('\n')
        for key in sorted(self.parametersAE):
            out.write(key + ": " + str(self.parameters[key]))
            out.write('\n')
        out.write('\n')
        out.write('Scores')
        out.write('\n')
        out.write("=" * 25)
        out.write('\n')
        for metric, score in metric_scores.items():
            out.write(metric + ': ' + str(score))
            out.write('\n')
        out.close()
        print("Report file successfully written.")

    def create_DNN_model(self, print_model=True):
        print("Creating DNN model")
        fundamental_parameters = ['dropout', 'output_activation', 'optimization', 'learning_rate',
                                  'units_in_input_layer',
                                  'units_in_hidden_layers', 'nb_epoch', 'batch_size']
        for param in fundamental_parameters:
            if self.parametersDNN[param] == None:
                print("Parameter not set: " + param)
                return
        self.print_parameter_values()
        model = Sequential()
        # Input layer - encoded data
        model.add(self.encoder)
        model.add(Dropout(self.parametersDNN['dropout']))
        # constructing all hidden layers
        for layer in self.parametersDNN['units_in_hidden_layers']:
            model.add(Dense(layer, activation='relu'))
            model.add(Dropout(self.parametersDNN['dropout']))
        # constructing the final layer
        model.add(Dense(1))
        model.add(Activation(self.parametersDNN['output_activation']))
        if self.parametersDNN['optimization'] == 'SGD':
            optim = SGD()
            optim.lr.set_value(self.parametersDNN['learning_rate'])
        elif self.parametersDNN['optimization'] == 'RMSprop':
            optim = RMSprop()
            optim.lr.set_value(self.parametersDNN['learning_rate'])
        elif self.parametersDNN['optimization'] == 'Adam':
            optim = Adam()
        elif self.parametersDNN['optimization'] == 'Adadelta':
            optim = Adadelta()
        model.compile(loss='binary_crossentropy', optimizer=optim, metrics=[matthews_correlation])
        if print_model: str(model.summary())
        self.model = model
        print("DNN model successfully created")


    def cv_fit(self, X, y, cv=5):
        if self.parametersDNN['nb_epoch'] and self.parametersDNN['batch_size']:
            self.create_DNN_model()
            init_weights = self.model.get_weights()
            cvscores = []
            cvhistory = []
            time_fit = []
            i = 0
            skf = StratifiedKFold(n_splits=cv, shuffle=False)
            for train, valid in skf.split(X, y):
                print("Running Fold " + str(i + 1) + str("/") + str(cv))
                X_train, X_valid = X[train], X[valid]
                y_train, y_valid = y[train], y[valid]
                self.model.set_weights(init_weights)
                time_fit.append(self.fit_model(X_train, X_valid, y_train, y_valid))
                cvscores.append(self.evaluate_model(X_valid, y_valid))
                cvhistory.append(self.history)
                i += 1
        return cvscores, cvhistory, time_fit

    def fit_model(self, X_train, X_test, y_train, y_test):
        print("Fitting DNN model")
        start_time = time.time()
        if self.parametersDNN['nb_epoch'] and self.parametersDNN['batch_size']:
            if self.parametersDNN['early_stopping']:
                early_stopping = EarlyStopping(monitor='val_loss', patience=self.parametersDNN['patience'])
                self.history = self.model.fit(X_train, y_train, epochs=self.parametersDNN['nb_epoch'],
                                              batch_size=self.parametersDNN['batch_size'],
                                              verbose=self.verbose, validation_data=(X_test, y_test),
                                              callbacks=[early_stopping])
            else:
                self.history = self.model.fit(X_train, y_train, epochs=self.parametersDNN['nb_epoch'],
                                              batch_size=self.parametersDNN['batch_size'],
                                              verbose=self.verbose, validation_data=(X_test, y_test))
        fit_time = time.time() - start_time
        print("DNN model successfully fit in ", timer(fit_time))
        return fit_time

    def print_fit_results(self, train_scores, val_scores):
        print('val_matthews_correlation: ', val_scores[1])
        print('val_loss: ', val_scores[0])
        print('train_matthews_correlation: ', train_scores[1])
        print('train_loss: ', train_scores[0])
        print("train/val loss ratio: ", min(self.history.history['loss']) / min(self.history.history['val_loss']))

    def evaluate_model(self, X_test, y_test):
        print("Evaluating model with hold out test set.")
        y_pred = self.model.predict(X_test)
        print(y_pred)
        y_pred = [float(np.round(x)) for x in y_pred]
        y_pred = np.ravel(y_pred)
        print(y_pred)
        print(y_test)
        scores = dict()
        scores['roc_auc'] = roc_auc_score(y_test, y_pred)
        scores['accuracy'] = accuracy_score(y_test, y_pred)
        scores['f1_score'] = f1_score(y_test, y_pred)
        scores['mcc'] = matthews_corrcoef(y_test, y_pred)
        scores['precision'] = precision_score(y_test, y_pred)
        scores['recall'] = recall_score(y_test, y_pred)
        scores['log_loss'] = log_loss(y_test, y_pred)
        for metric, score in scores.items():
            print(metric + ': ' + str(score))
        return scores

    def format_scores_cv(self, scores_cv_list):
        raw_scores = dict.fromkeys(list(scores_cv_list[0].keys()))
        for key, value in raw_scores.items():
            raw_scores[key] = []
        for score in scores_cv_list:
            for metric, value in score.items():
                raw_scores[metric].append(value)
        mean_scores = dict.fromkeys(list(scores_cv_list[0].keys()))
        sd_scores = dict.fromkeys(list(scores_cv_list[0].keys()))
        for metric in raw_scores.keys():
            mean_scores[metric] = np.mean(raw_scores[metric])
            sd_scores[metric] = np.std(raw_scores[metric])
        for metric in mean_scores.keys():
            print(metric, ': ', str(mean_scores[metric]), ' +/- ', sd_scores[metric])
        return mean_scores, sd_scores, raw_scores

    def model_selection(self, X, y, n_iter_search=2, n_folds=2):
        print("Selecting best DNN model")
        old_parameters = self.parameters.copy()
        self.model_selection_history = []
        for iteration in range(n_iter_search):
            mean_train_matthews_correlation = []
            mean_train_loss = []
            mean_val_matthews_correlation = []
            mean_val_loss = []
            mean_time_fit = []
            print("Iteration no. " + str(iteration + 1))
            new_parameters = self.batch_parameter_shufller()
            temp_values = new_parameters.copy()
            for key in new_parameters:
                self.parameters[key] = new_parameters[key]
            i = 0
            skf = StratifiedKFold(n_splits=n_folds, shuffle=False,)
            for train, valid in skf.split(X, y):
                print("Running Fold " + str(i + 1) + str("/") + str(n_folds))
                X_train, X_valid = X[train], X[valid]
                y_train, y_valid = y[train], y[valid]
                self.create_DNN_model()
                time_fit = self.fit_model(X_train, X_valid, y_train, y_valid)
                train_scores = self.model.evaluate(X_train, y_train)
                train_matthews_correlation = train_scores[1]
                train_loss = train_scores[0]
                val_scores = self.model.evaluate(X_valid, y_valid)
                val_matthews_correlation = val_scores[1]
                val_loss = val_scores[0]
                self.print_fit_results(train_scores, val_scores)
                mean_train_matthews_correlation.append(train_matthews_correlation)
                mean_train_loss.append(train_loss)
                mean_val_matthews_correlation.append(val_matthews_correlation)
                mean_val_loss.append(val_loss)
                mean_time_fit.append(time_fit)
                temp_values['train_matthews_correlation_' + str(i + 1)] = train_matthews_correlation
                temp_values['train_loss_' + str(i + 1)] = train_loss
                temp_values['val_matthews_correlation_' + str(i + 1)] = val_matthews_correlation
                temp_values['val_loss_' + str(i + 1)] = val_loss
                temp_values['time_fit_' + str(i + 1)] = time_fit
                i += 1
            temp_values['mean_train_matthews_correlation'] = np.mean(mean_train_matthews_correlation)
            temp_values['mean_train_loss'] = np.mean(mean_train_loss)
            temp_values['mean_val_matthews_correlation'] = np.mean(mean_val_matthews_correlation)
            temp_values['mean_val_loss'] = np.mean(mean_val_loss)
            temp_values['mean_time_fit'] = np.mean(mean_time_fit)
            self.model_selection_history.append(temp_values)
        self.parameters = old_parameters.copy()
        print("Best DNN model successfully selected")

    def find_best_model(self):
        if self.model_selection_history:
            best_model = None
            max_val_matthews_correlation = 0
            for dic in self.model_selection_history:
                if dic['mean_val_matthews_correlation'] > max_val_matthews_correlation:
                    best_model = dic
                    max_val_matthews_correlation = dic['mean_val_matthews_correlation']
        # If all models have score 0 assume the first one
        if best_model is None:
            best_model = self.model_selection_history[0]
        print("Best model:")
        self.print_parameter_values()
        print("Matthews correlation: " + str(max_val_matthews_correlation))
        return best_model



    def set_filename(self):
        self.filename = HighThroughput.endpoint_name

    def plot_model_performance(self, cv_history, root_dir, file_name, save_fig=True, show_plot=False):
        # summarize history for loss
        ## Plotting the loss with the number of iterations
        fig = plt.figure(figsize=(20, 15))
        fig.add_subplot(121)
        for record in cv_history:
            plt.semilogy(record.history['loss'], color='blue')
            plt.semilogy(record.history['val_loss'], color='orange')
        plt.legend(['train', 'test'], loc='upper left')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        ## Plotting the error with the number of iterations
        ## With each iteration the error reduces smoothly
        fig.add_subplot(122)
        for record in cv_history:
            plt.plot(record.history['matthews_correlation'], color='blue')
            plt.plot(record.history['val_matthews_correlation'], color='orange')
            plt.legend(['train', 'test'], loc='upper left')
        plt.title('model matthews correlation')
        plt.ylabel('matthews correlation')
        plt.xlabel('epoch')
        if save_fig:
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)
            i = 0
            while os.path.exists(root_dir + '/' + file_name + '_graph_results_' + str(i) + '.png'):
                i += 1
            file_name = os.path.join(root_dir, file_name + '_graph_results_' + str(i) + '.png')
            print("Writing graph results file with path: ", file_name)
            plt.savefig(file_name)
        if show_plot: plt.show()

    def write_model_selection_results(self, root_dir, file_name):
        d = self.model_selection_history
        sequence = ['mean_val_matthews_correlation', 'mean_val_loss', 'mean_train_matthews_correlation',
                    'mean_train_loss', 'mean_time_fit']
        sequence_tv_scores = [key for key in d[0] if key.startswith(("train_", "val_", "time_"))]
        sequence_tv_scores.sort()
        sequence_parameters = [x for x in self.parameters]
        sequence_parameters.sort()
        sequence.extend(sequence_tv_scores + sequence_parameters)
        df = pd.DataFrame(d, columns=sequence).sort_values(['mean_val_matthews_correlation'], ascending=[False])
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        i = 0
        while os.path.exists(root_dir + '/' + file_name + '_model_selection_table_' + str(i) + '.csv'):
            i += 1
        final_path = os.path.join(root_dir, file_name + '_model_selection_table_' + str(i) + '.csv')
        print('Writing csv file with path: ', final_path)
        df.to_csv(final_path, sep='\t')
        self.model_selection_results = df

    def write_report(self, mean_scores, sd_scores, raw_scores, root_dir, file_name):
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        i = 0
        while os.path.exists(root_dir + '/' + file_name + '_report_' + str(i) + '.txt'):
            i += 1
        final_path = os.path.join(root_dir, file_name + '_report_' + str(i) + '.txt')
        print("Writing report file with path: " + final_path)
        out = open(final_path, 'w')
        out.write('Stacked Autoencoder')
        out.write('\n')
        out.write('=' * 25)
        out.write('\n')
        out.write('Hyperparameters')
        out.write('\n')
        out.write('=' * 25)
        out.write('\n')
        for key in sorted(self.parametersAE):
            out.write(key + ": " + str(self.parametersAE[key]))
            out.write('\n')
        out.write('\n')
        out.write('DNN')
        out.write('\n')
        out.write('=' * 25)
        out.write('\n')
        out.write('Hyperparameters')
        out.write('\n')
        out.write('=' * 25)
        out.write('\n')
        for key in sorted(self.parametersDNN):
            out.write(key + ": " + str(self.parametersDNN[key]))
            out.write('\n')
        out.write('\n')
        out.write('Scores')
        out.write('\n')
        out.write("=" * 25)
        out.write('\n')
        for metric, scores in mean_scores.items():
            out.write(str(metric) + ': ' + str(mean_scores[metric]) + ' +/- ' + str(sd_scores[metric]))
            out.write('\n')
        out.close()
        df = pd.DataFrame.from_dict(raw_scores)
        cv_df_path = os.path.join(root_dir, file_name + '_cv_results_' + str(i) + '.csv')
        print('Writing csv file with path: ', cv_df_path)
        df.to_csv(cv_df_path, sep='\t')
        print("Report files successfully written.")


    def select_best_model(self):
        best_model = self.find_best_model()
        for key in self.parametersDNN:
            self.parametersDNN[key] = best_model[key]

    def print_parameter_values(self):
        print("Hyperparameters")
        for key in sorted(self.parametersDNN):
            print(key + ": " + str(self.parametersDNN[key]))





def model_selection_all(n_iter, c_val, verbose=1):
    rnaseq = HighThroughput('GSE49711_SEQC_NB_MAV_G_log2.20121127.txt', 'clinicaldata.csv')
    rnaseq.load_data()
    rnaseq.variance_filter(10)
    list_enpoints = ['endpoint_Sex', 'endpoint_EFSAll', 'endpoint_ClassLabel', 'endpoint_OSAll', 'endpoint_OSHR',
                     'endpoint_EFSHR']
    for endpoint in list_enpoints:
        function_call = getattr(rnaseq, endpoint)
        X, y = function_call()
        dnn = SDAEModel(X, y)
        dnn.verbose = verbose
        dnn.filename = rnaseq.endpoint_name
        dnn.model_selection_AE(n_iter, c_val)
        dnn.write_model_selection_results_AE()
        dnn.select_best_model_AE()
        dnn.create_DNN_model()
        dnn.fit_SDAE_model()
        dnn.print_fit_results()
        dnn.plot_model_performance()
        dnn.write_report_AE()


def model_selection_endpoint(endpoint_name, n_iter, c_val, verbose=1):
    rnaseq = HighThroughput('GSE49711_SEQC_NB_MAV_G_log2.20121127.txt', 'clinicaldata.csv')
    rnaseq.load_data()
    function_call = getattr(rnaseq, endpoint_name)
    X, y = function_call()
    dnn = SDAEModel(X, y)
    dnn.normalization()
    dnn.train_valid_split()
    dnn.filename = rnaseq.endpoint_name
    dnn.model_selection_AE(n_iter, c_val)
    dnn.write_model_selection_results_AE()
    dnn.select_best_model_AE()
    dnn.create_DNN_model()
    dnn.fit_SDAE_model()
    dnn.print_fit_results()
    dnn.plot_model_performance()
    dnn.write_report_AE()
    dnn.kfold_cv_fit()


def model_fit(dropout, output_activation, optimization, learning_rate, units_in_input_layer,
              units_in_hidden_layers, nb_epoch, batch_size, early_stopping, patience, file_name, cv, X, y):
    dnn = SDAEModel(X=X, y=y, cv=cv)
    dnn.parameters = {

        'dropout': dropout,
        'output_activation': output_activation,
        'optimization': optimization,
        'learning_rate': learning_rate,
        'units_in_input_layer': units_in_input_layer,
        'units_in_hidden_layers': units_in_hidden_layers,
        'nb_epoch': nb_epoch,
        'batch_size': batch_size,
        'early_stopping': early_stopping,
        'patience': patience

    }
    dnn.create_SDAE_model()
    dnn.fit_SDAE_model(X)
    dnn.write_report_AE()


def create_SDAE(endpoint_name, n_iter, verbose=0):
    rnaseq = HighThroughput_MMDC('MMRF_CoMMpass_IA9_E74GTF_Salmon_Gene_TPM.txt', 'globalClinTraining.csv')
    # rnaseq = HighThroughput('GSE49711_SEQC_NB_MAV_G_log2.20121127.txt','clinicaldata.csv')
    rnaseq.load_data()
    # rnaseq.variance_filter(10)
    function_call = getattr(rnaseq, endpoint_name)
    X, y = function_call()
    dnn = SDAEModel(X, y)
    dnn.normalization()
    dnn.train_valid_split()
    dnn.filename = rnaseq.endpoint_name
    dnn.create_SDAE_model()
    dnn.fit_SDAE_model(dnn.X_train, dnn.X_valid)
    dnn.print_fit_results()
    dnn.create_DNN_model()
    dnn.fit_SDAE_model()
    dnn.print_fit_results()


def model_selection_all_AE(n_iter, c_val, verbose=0):
    rnaseq = HighThroughput('GSE49711_SEQC_NB_MAV_G_log2.20121127.txt', 'clinicaldata.csv')
    rnaseq.load_data()
    # rnaseq.variance_filter(5)
    list_enpoints = ['endpoint_Sex', 'endpoint_EFSAll', 'endpoint_ClassLabel', 'endpoint_OSAll', 'endpoint_OSHR',
                     'endpoint_EFSHR']
    for endpoint in list_enpoints:
        function_call = getattr(rnaseq, endpoint)
        X, y = function_call()
        dnn = SDAEModel(X, y)
        dnn.normalization()
        dnn.filename = rnaseq.endpoint_name
        dnn.train_valid_split()
        dnn.model_selection_AE(dnn.X_train, dnn.y_train, n_iter, c_val)
        dnn.write_model_selection_results_AE()
        dnn.select_best_model_AE()
        dnn.create_SDAE_model()
        dnn.fit_SDAE_model(dnn.X_train, dnn.X_valid)
        dnn.print_fit_results()


def model_selection_endpoint_AE(endpoint_name, n_iter, c_val, verbose=0):
    rnaseq = HighThroughput('GSE49711_SEQC_NB_MAV_G_log2.20121127.txt', 'clinicaldata.csv')
    rnaseq.load_data()
    # rnaseq.variance_filter(10)
    function_call = getattr(rnaseq, endpoint_name)
    X, y = function_call()
    dnn = SDAEModel(X, y)
    dnn.normalization()
    dnn.train_valid_split()
    dnn.filename = rnaseq.endpoint_name
    dnn.model_selection_AE(dnn.X_train, dnn.y_train, n_iter, c_val)
    dnn.write_model_selection_results_AE()
    dnn.select_best_model_AE()
    dnn.create_SDAE_model()
    dnn.fit_SDAE_model(dnn.X_train, dnn.X_valid)
    dnn.print_fit_results()


def model_selection_NB(n_iter, c_val):
    start_time = time.time()
    list_features = [5000]  # [100,250,500,1000,2000,5000]
    list_endpoints = ['endpoint_Sex', 'endpoint_EFSAll', 'endpoint_ClassLabel', 'endpoint_OSAll', 'endpoint_OSHR',
                      'endpoint_EFSHR']
    list_fsel = ['f_classif']
    # autoencoder
    ae_root = os.path.join('Endpoints_Xy', 'endpoint_Sex')
    ae_filename = '_NB_5000_f_classif.csv'
    X_train_name = os.path.join(ae_root, 'X_train' + ae_filename)
    X_test_name = os.path.join(ae_root, 'X_test' + ae_filename)
    y_train_name = os.path.join(ae_root, 'y_train' + ae_filename)
    y_test_name = os.path.join(ae_root, 'y_test' + ae_filename)
    X_train = np.genfromtxt(X_train_name)
    y_train = np.genfromtxt(y_train_name)
    X_test = np.genfromtxt(X_test_name)
    y_test = np.genfromtxt(y_test_name)
    sdae = SDAEModel(X_train, y_train, X_test, y_test)
    sdae.filename = 'sdae_model_selection'
    sdae.model_selection_AE(sdae.X_train, sdae.y_train, n_iter, c_val)
    sdae.write_model_selection_results_AE()
    sdae.select_best_model_AE()
    sdae.create_SDAE_model()
    sdae.fit_SDAE_model(sdae.X_train, sdae.X_test)
    sdae.print_fit_results()

    # for endpoint in list_endpoints:
    # 	root = os.path.join('Endpoints_Xy', endpoint)
    # 	for features in list_features:
    # 		for fsel in list_fsel:
    # 			file_name = '_NB_' + str(features) + '_' + fsel + '.csv'
    # 			X_train_name = os.path.join(root, 'X_train' + file_name)
    # 			X_test_name = os.path.join(root, 'X_test' + file_name)
    # 			y_train_name = os.path.join(root, 'y_train' + file_name)
    # 			y_test_name = os.path.join(root, 'y_test' + file_name)
    # 			X_train = np.genfromtxt(X_train_name)
    # 			y_train = np.genfromtxt(y_train_name)
    # 			X_test = np.genfromtxt(X_test_name)
    # 			y_test = np.genfromtxt(y_test_name)
    # 			dnn = Model(X_train, y_train, X_test, y_test)
    # 			dnn.endpoint_name = endpoint
    # 			dnn.filename = str(features) + '_' + fsel
    # 			dnn.model_selection(dnn.X_train, dnn.y_train, n_iter, c_val)
    # 			dnn.write_model_selection_results()
    # 			dnn.select_best_model()
    # 			dnn.create_DNN_model()
    # 			dnn.fit_model(dnn.X_train, dnn.X_test, dnn.y_train, dnn.y_test)
    # 			dnn.print_fit_results()
    # 			dnn.plot_model_performance()
    # 			scores = dnn.evaluate_model(dnn.X_test, dnn.y_test)
    # 			dnn.write_report(scores)
    end_time = time.time()
    print("Script run in ", timer(end_time - start_time))

def sdae_model_fit(output_activation, optimization, learning_rate, units_in_input_layer,
                  units_in_hidden_layers, nb_epoch, batch_size, early_stopping, patience, root_dir, file_name, cv,
                  X, y):
    sdae = SDAEModel(X=X, y=y, cv=cv)
    sdae.parametersAE = {

        'output_activation': output_activation,
        'optimization': optimization,
        'learning_rate': learning_rate,
        'units_in_input_layer': units_in_input_layer,
        'units_in_hidden_layers': units_in_hidden_layers,
        'nb_epoch': nb_epoch,
        'batch_size': batch_size,
        'early_stopping': early_stopping,
        'patience': patience

    }
    sdae.create_SDAE_model()
    # sdae.model_selection(sdae.X, sdae.y, 2, sdae.cv)
    # sdae.find_best_model()
    # sdae.select_best_model()
    # sdae.create_SDAE_model()
    sdae.fit_SDAE_model(sdae.X)
    sdae.create_DNN_model()
    cv_scores, cv_history, time_fit = sdae.cv_fit(sdae.X, sdae.y, cv)
    sdae.plot_model_performance(cv_history, root_dir, file_name)
    mean_scores, sd_scores, raw_scores = sdae.format_scores_cv(cv_scores)
    sdae.write_report(mean_scores, sd_scores, raw_scores, root_dir, file_name)

def dnn_model_selection_split(X_train, X_test, y_train, y_test, root_dir, experiment_designation, n_iter=10, cv=5):
    dnn = DNNModel(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, cv=cv)
    file_name = experiment_designation
    dnn.model_selection_AE(dnn.X_train, dnn.y_train, n_iter, cv)
    dnn.write_model_selection_results_AE(root_dir, file_name)
    dnn.select_best_model_AE()
    dnn.create_DNN_model()
    cv_scores, cv_history, time_fit = dnn.cv_fit(dnn.X_test, dnn.y_test, cv)
    dnn.plot_model_performance(cv_history, root_dir, file_name)
    mean_scores, sd_scores, raw_scores = dnn.format_scores_cv(cv_scores)
    dnn.write_report_AE(mean_scores, sd_scores, raw_scores, root_dir, file_name)


def dnn_model_selection_cv(X, y, root_dir, experiment_designation, n_iter=100, cv=10):
    dnn = DNNModel(X=X, y=y, cv=cv)
    file_name = experiment_designation
    dnn.model_selection_AE(dnn.X, dnn.y, n_iter, cv)
    dnn.write_model_selection_results_AE(root_dir, file_name)
    dnn.select_best_model_AE()
    dnn.create_DNN_model()
    cv_scores, cv_history, time_fit = dnn.cv_fit(dnn.X, dnn.y, cv)
    dnn.plot_model_performance(cv_history, root_dir, file_name)
    mean_scores, sd_scores, raw_scores = dnn.format_scores_cv(cv_scores)
    dnn.write_report_AE(mean_scores, sd_scores, raw_scores, root_dir, file_name)
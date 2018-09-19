#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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


class DNNModel:
    def __init__(self, **kwargs):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.splitted = None
        self.feature_number = None
        # parameters of selected DNN model
        self.parameters = {
            'dropout': 0.1,
            'output_activation': 'sigmoid',
            'optimization': 'SGD',
            'learning_rate': 0.005,
            'units_in_input_layer': 5000,
            'units_in_hidden_layers': [2500, 250, 10],
            'nb_epoch': 400,
            'batch_size': 75,
            'early_stopping': True,
            'patience': 30
        }
        self.filename = None
        self.verbose = 0
        # model selection parameters
        self.parameters_batch = {
            'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'output_activation': ['sigmoid'],
            'optimization': ['SGD', 'Adam', 'RMSprop'],
            'learning_rate': [0.015, 0.010, 0.005, 0.001],
            'batch_size': [60, 75, 80, 100, 120, 150],
            'nb_epoch': [1000],
            'units_in_hidden_layers': [[2500, 1000, 500], [1000, 100], [2500, 1000, 500, 100], [2500, 100, 10],
                                       [2500, 100], [2500, 500]],
            'units_in_input_layer': [5000],
            'early_stopping': [True],
            'patience': [80]
        }
        self.model_selection_history = []
        if validate_matrices(kwargs):
            if len(kwargs.keys()) <= 3:
                self.X = kwargs['X']
                self.y = kwargs['y']
                self.splitted = False
            elif len(kwargs.keys()) >= 4:
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

    def print_parameter_values(self):
        print("Hyperparameters")
        for key in sorted(self.parameters):
            print(key + ": " + str(self.parameters[key]))

    def create_DNN_model(self, verbose=True):
        print("Creating DNN model")
        fundamental_parameters = ['dropout', 'output_activation', 'optimization', 'learning_rate',
                                  'units_in_input_layer',
                                  'units_in_hidden_layers', 'nb_epoch', 'batch_size']
        for param in fundamental_parameters:
            if self.parameters[param] == None:
                print("Parameter not set: " + param)
                return
        self.print_parameter_values()
        model = Sequential()
        # Input layer
        model.add(Dense(self.parameters['units_in_input_layer'], input_dim=self.feature_number, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.parameters['dropout']))
        # constructing all hidden layers
        for layer in self.parameters['units_in_hidden_layers']:
            model.add(Dense(layer, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(self.parameters['dropout']))
        # constructing the final layer
        model.add(Dense(1, activation=self.parameters['output_activation']))
        if self.parameters['optimization'] == 'SGD':
            optim = SGD()
            optim.lr.set_value(self.parameters['learning_rate'])
        elif self.parameters['optimization'] == 'RMSprop':
            optim = RMSprop()
            optim.lr.set_value(self.parameters['learning_rate'])
        elif self.parameters['optimization'] == 'Adam':
            optim = Adam()
        elif self.parameters['optimization'] == 'Adadelta':
            optim = Adadelta()
        model.compile(loss='binary_crossentropy', optimizer=optim, metrics=[matthews_correlation])
        if self.verbose == 1: str(model.summary())
        self.model = model
        print("DNN model sucessfully created")

    def cv_fit(self, X, y, cv=5):
        if self.parameters['nb_epoch'] and self.parameters['batch_size']:
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
        if self.parameters['nb_epoch'] and self.parameters['batch_size']:
            if self.parameters['early_stopping']:
                early_stopping = EarlyStopping(monitor='val_loss', patience=self.parameters['patience'])
                self.history = self.model.fit(X_train, y_train, epochs=self.parameters['nb_epoch'],
                                              batch_size=self.parameters['batch_size'],
                                              verbose=self.verbose, validation_data=(X_test, y_test),
                                              callbacks=[early_stopping])
            else:
                self.history = self.model.fit(X_train, y_train, epochs=self.parameters['nb_epoch'],
                                              batch_size=self.parameters['batch_size'],
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

    def select_best_model(self):
        best_model = self.find_best_model()
        for key in self.parameters:
            self.parameters[key] = best_model[key]

    def batch_parameter_shufller(self):
        chosen_param = {}
        for key in self.parameters_batch:
            chosen_param[key] = choice(self.parameters_batch[key])
        return chosen_param

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
        out.write('Hyperparameters')
        out.write('\n')
        out.write('=' * 25)
        out.write('\n')
        for key in sorted(self.parameters):
            out.write(key + ": " + str(self.parameters[key]))
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


def dnn_model_fit(dropout, output_activation, optimization, learning_rate, units_in_input_layer,
                  units_in_hidden_layers, nb_epoch, batch_size, early_stopping, patience, root_dir, file_name, cv,
                  X, y):
    dnn = DNNModel(X=X, y=y, cv=cv)
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
    cv_scores, cv_history, time_fit = dnn.cv_fit(dnn.X, dnn.y, cv)
    dnn.plot_model_performance(cv_history, root_dir, file_name)
    mean_scores, sd_scores, raw_scores = dnn.format_scores_cv(cv_scores)
    dnn.write_report(mean_scores, sd_scores, raw_scores, root_dir, file_name)
    time_fit = np.asarray(time_fit, dtype=float)
    del dnn.model


def dnn_model_selection_split(X_train, X_test, y_train, y_test, root_dir, experiment_designation, n_iter=10, cv=5):
    dnn = DNNModel(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, cv=cv)
    file_name = experiment_designation
    dnn.model_selection(dnn.X_train, dnn.y_train, n_iter, cv)
    dnn.write_model_selection_results(root_dir, file_name)
    dnn.select_best_model()
    dnn.create_DNN_model()
    cv_scores, cv_history, time_fit = dnn.cv_fit(dnn.X_test, dnn.y_test, cv)
    dnn.plot_model_performance(cv_history, root_dir, file_name)
    mean_scores, sd_scores, raw_scores = dnn.format_scores_cv(cv_scores)
    dnn.write_report(mean_scores, sd_scores, raw_scores, root_dir, file_name)


def dnn_model_selection_cv(X, y, root_dir, experiment_designation, n_iter=100, cv=10):
    dnn = DNNModel(X=X, y=y, cv=cv)
    file_name = experiment_designation
    dnn.model_selection(dnn.X, dnn.y, n_iter, cv)
    dnn.write_model_selection_results(root_dir, file_name)
    dnn.select_best_model()
    dnn.create_DNN_model()
    cv_scores, cv_history, time_fit = dnn.cv_fit(dnn.X, dnn.y, cv)
    dnn.plot_model_performance(cv_history, root_dir, file_name)
    mean_scores, sd_scores, raw_scores = dnn.format_scores_cv(cv_scores)
    dnn.write_report(mean_scores, sd_scores, raw_scores, root_dir, file_name)

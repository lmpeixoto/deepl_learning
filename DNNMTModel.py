#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# Imports
from sklearn.model_selection import KFold
from DNNModel import *
from numpy import *
from external_functions import *
import matplotlib
matplotlib.use('Agg')
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, RMSprop, Adadelta, Adam




class DNNMTModel(DNNModel):


    def create_DNN_model(self, print_model=True):
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
        model.add(Dropout(self.parameters['dropout']))
        # constructing all hidden layers
        for layer in self.parameters['units_in_hidden_layers']:
            model.add(Dense(layer, activation='relu'))
            model.add(Dropout(self.parameters['dropout']))
        # constructing the final layer - Multi task DNN - 6 endpoints predicts the outcome given an input X.
        model.add(Dense(6, activation=self.parameters['output_activation']))
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


    def model_selection(self, X, y, n_iter_search=2, n_folds=2):
        print("Selecting best DNN model")
        seed = 7
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
            kf = KFold(n_splits=n_folds, shuffle=False, random_state=seed)
            for train, valid in kf.split(X, y):
                print("Running Fold " + str(i + 1) + str("/") + str(n_folds))
                X_train, X_valid = X[train], X[valid]
                y_train, y_valid = y[train], y[valid]
                self.create_DNN_model()
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

    def cv_fit(self, X, y, cv=5):
        if self.parameters['nb_epoch'] and self.parameters['batch_size']:
            cvscores = []
            cvhistory = []
            i = 0
            kf = KFold(n_splits=cv, shuffle=False, random_state=1)
            for train, valid in kf.split(X, y):
                print("Running Fold " + str(i + 1) + str("/") + str(cv))
                X_train, X_valid = X[train], X[valid]
                y_train, y_valid = y[train], y[valid]
                self.create_DNN_model()
                time_fit = self.fit_model(X_train, X_valid, y_train, y_valid)
                cvscores.append(self.evaluate_model(X_valid, y_valid))
                cvhistory.append(self.history)
                i += 1
        return cvscores, cvhistory

    def evaluate_model(self, X_test, y_test):
        print("Evaluating model with hold out test set.")
        scores = dict()
        endpoints = ['endpoint_Sex', 'endpoint_EFSAll', 'endpoint_ClassLabel', 'endpoint_OSAll', 'endpoint_OSHR',
                     'endpoint_EFSHR']
        predictions = self.model.predict(X_test).round().astype(int)
        print(predictions)
        print(y_test)
        scores['roc_auc'] = []
        scores['accuracy'] = []
        scores['f1_score'] = []
        scores['mcc'] = []
        scores['precision'] = []
        scores['recall'] = []
        scores['log_loss'] = []
        for column in range(predictions.shape[1]):
            y_true = y_test[:, column]
            y_pred = predictions[:, column]
            scores['roc_auc'].append(roc_auc_score(y_true, y_pred))
            scores['accuracy'].append(accuracy_score(y_true, y_pred))
            scores['f1_score'].append(f1_score(y_true, y_pred))
            scores['mcc'].append(matthews_corrcoef(y_true, y_pred))
            scores['precision'].append(precision_score(y_true, y_pred))
            scores['recall'].append(recall_score(y_true, y_pred))
            scores['log_loss'].append(log_loss(y_true, y_pred))
        for i in range(len(endpoints)):
            print(endpoints[i])
            for metric, score in scores.items():
                print(metric + ': ' + str(score[i]))
        return scores


    def format_scores_cv(self, scores_cv_list):
        form_scores = dict()
        for key in scores_cv_list[0].keys():
            form_scores[key] = []
        for score in scores_cv_list:
            form_scores['roc_auc'].append(score['roc_auc'])
            form_scores['accuracy'].append(score['accuracy'])
            form_scores['f1_score'].append(score['f1_score'])
            form_scores['mcc'].append(score['mcc'])
            form_scores['precision'].append(score['precision'])
            form_scores['recall'].append(score['recall'])
            form_scores['log_loss'].append(score['log_loss'])
        mean_scores = dict()
        sd_scores = dict()
        for key in form_scores.keys():
            mean_scores[key] = []
            sd_scores[key] = []
        for key, value in mean_scores.items():
            mean_scores[key] = [[] for i in range(6)]
            sd_scores[key] = [[] for i in range(6)]
        for endpoint in range(6):

            for metric in form_scores.keys():
                temp_cv_array = []
                for fold in range(self.cv):
                    temp_cv_array.append(form_scores[metric][fold][endpoint])
                mean_scores[metric][endpoint] = np.mean(temp_cv_array)
                sd_scores[metric][endpoint] = np.std(temp_cv_array)
            for metric in mean_scores.keys():
                print(metric, ': ', str(mean_scores[metric][endpoint]), ' +/- ', sd_scores[metric][endpoint])
        return mean_scores, sd_scores


    def write_report(self, mean_scores, sd_scores, root_dir, file_name):
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        i = 0
        while os.path.exists(root_dir + '/' + file_name + '_report_' + str(i) + '.txt'):
            i += 1
        final_path = os.path.join(root_dir, file_name + '_report_' + str(i) + '.txt')
        print("Writing report file with path: " + final_path)
        endpoints = ['endpoint_Sex', 'endpoint_EFSAll', 'endpoint_ClassLabel', 'endpoint_OSAll', 'endpoint_OSHR',
                     'endpoint_EFSHR']
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
        for i in range(len(endpoints)):
            out.write('\n')
            out.write(endpoints[i])
            out.write('\n')
            for metric in mean_scores.keys():
                out.write(str(metric) + ': ' + str(mean_scores[metric][i]) + ' +/- ' + str(sd_scores[metric][i]))
                out.write('\n')
        out.close()
        print("Report file successfully written.")

def dnn_mt_model_fit(dropout, output_activation, optimization, learning_rate, units_in_input_layer,
                  units_in_hidden_layers, nb_epoch, batch_size, early_stopping, patience, root_dir, file_name, cv=5,
                  **kwargs):
    dnn = DNNMTModel(kwargs)
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
    dnn.create_DNN_model()
    if not dnn.splitted:
        cv_scores, cv_history = dnn.cv_fit(dnn.X, dnn.y, cv)
        dnn.print_fit_results()
        dnn.plot_model_performance(cv_history, root_dir, file_name)
        mean_scores, sd_scores = dnn.format_scores_cv(cv_scores)
        dnn.write_report(mean_scores, sd_scores, root_dir, file_name)
    else:
        cv_scores, cv_history = dnn.cv_fit(dnn.X_test, dnn.y_test, cv)
        dnn.print_fit_results()
        dnn.plot_model_performance(cv_history, root_dir, file_name)
        mean_scores, sd_scores = dnn.format_scores_cv(cv_scores)
        dnn.write_report(mean_scores, sd_scores, root_dir, file_name)


def dnn_mt_model_selection_split(X_train, X_test, y_train, y_test, root_dir, experiment_designation, n_iter=10, cv=5):
    dnn = DNNMTModel(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, cv=cv)
    file_name = experiment_designation
    dnn.model_selection(dnn.X_train, dnn.y_train, n_iter, cv)
    dnn.write_model_selection_results(root_dir, file_name)
    dnn.select_best_model()
    dnn.create_DNN_model()
    cv_scores, cv_history = dnn.cv_fit(dnn.X_test, dnn.y_test, cv)
    dnn.plot_model_performance(cv_history, root_dir, file_name)
    mean_scores, sd_scores = dnn.format_scores_cv(cv_scores)
    dnn.write_report(mean_scores, sd_scores, root_dir, file_name)


def dnn_mt_model_selection_cv(X, y, root_dir, experiment_designation, n_iter=10, cv=5):
    dnn = DNNMTModel(X=X, y=y, cv=cv)
    file_name = experiment_designation
    dnn.model_selection(dnn.X, dnn.y, n_iter, cv)
    dnn.write_model_selection_results(root_dir, file_name)
    dnn.select_best_model()
    dnn.create_DNN_model()
    cv_scores, cv_history = dnn.cv_fit(dnn.X, dnn.y, cv)
    dnn.plot_model_performance(cv_history, root_dir, file_name)
    mean_scores, sd_scores = dnn.format_scores_cv(cv_scores)
    dnn.write_report(mean_scores, sd_scores, root_dir, file_name)



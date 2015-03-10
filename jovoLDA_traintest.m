function [labels_pred, n_errors, classifier, labels_prob] = jovoLDA_traintest( data_train, labels_train, data_test, labels_test, Opts )

% LDA train and test

if nargin<5,                    Opts = [];          end;
if ~isfield(Opts,'Priors'),     Opts.Priors = [];   end;

% [Yhat, eta, parms] = LDA_train_and_predict(Xtrain, Ytrain, Xtest)
[labels_pred, n_errors, labels_prob, classifier] = LDA_train_and_predict(data_test', data_train', labels_train');

% INPUT:
%   X in R^{D x n}: predictor matrix
%   Y in {0,1,NaN}^n: predictee matrix
end

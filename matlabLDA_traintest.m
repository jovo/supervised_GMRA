function [labels_pred, n_errors, classifier, labels_prob] = matlabLDA_traintest( data_train, labels_train, data_test, labels_test, Opts )

% LDA train and test

if nargin<5,                    Opts = [];          end;
if ~isfield(Opts,'Priors'),     Opts.Priors = [];   end;

% function [outclass, err, posterior, logp, coeffs] = classify(sample, training, group, type, prior)
[labels_pred, n_errors, labels_prob, logp, classifier] = classify(data_test', data_train', labels_train', 'linear', 'empirical');

end

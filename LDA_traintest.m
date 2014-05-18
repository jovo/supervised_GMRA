function [labels_pred, n_errors, classifier, labels_prob] = LDA_traintest( data_train, labels_train, data_test, labels_test, Opts )

% LDA train and test

if nargin<5,                    Opts = [];          end;
if ~isfield(Opts,'Priors'),     Opts.Priors = [];   end;

% Build LDA classifier
classifier                           = LDA_train(data_train', labels_train',Opts.Priors);

% Test LDA classifier
[n_errors, labels_pred, labels_prob] = LDA_test( classifier, data_test, labels_test );

end

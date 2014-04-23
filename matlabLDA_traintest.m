function [labels_pred, n_errors, classifier, labels_prob] = matlabLDA_traintest( data_train, labels_train, data_test, labels_test, Opts )

% LDA train and test

if nargin<5,                    Opts = [];          end;
if ~isfield(Opts,'Priors'),     Opts.Priors = [];   end;

% function [outclass, err, posterior, logp, coeffs] = classify(sample, training, group, type, prior)
[labels_pred, n_errors, labels_prob, logp, classifier_pre] = classify(data_test', data_train', labels_train', 'linear', 'empirical');
whos classifier_pre
whos data_test data_train labels_train
unique(labels_train)
if ~isfield(classifier_pre(1,2), 'const') % When there is only one unique label in the node
    classifier.W = [];
else
    classifier.W = [classifier_pre(1,2).const classifier_pre(1,2).linear'];
end

disp('debuggggggggggggggggggggging')
classifier.ClassLabel = unique(labels_train)
% classifier.labels_test = labels_test;

end

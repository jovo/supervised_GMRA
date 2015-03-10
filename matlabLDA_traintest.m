function [labels_pred, n_errors, classifier, labels_prob] = matlabLDA_traintest( data_train, labels_train, data_test, labels_test, Opts )

% LDA train and test

if nargin<5,                    Opts = [];          end;
if ~isfield(Opts,'Priors'),     Opts.Priors = [];   end;

% function [outclass, err, posterior, logp, coeffs] = classify(sample, training, group, type, prior)
[labels_pred, n_errors, labels_prob_pre, ~, classifier_pre] = classify(data_test', data_train', labels_train', 'linear', 'empirical');
  
% size(labels_prob)
% % Set the labels_prob same as how mauro's lda sets it.
% [size(labels_prob,2) size(labels_prob,1)]
% size(labels_pred')
% size(1:length(labels_pred))
% labels_prob = labels_prob(sub2ind([size(labels_prob,2) size(labels_prob,1)], labels_pred'+1, 1:length(labels_pred)))';
labels_pred = double(labels_pred);
labels_prob(labels_pred ==0) = labels_prob_pre(labels_pred == 0,1);
labels_prob(labels_pred ==1) = labels_prob_pre(labels_pred == 1,2);

if nargin>3 % If 'labels_test' is provided as input
    if ~isempty(labels_test)
        n_errors = sum(labels_pred ~= labels_test');
    else
        n_errors = [];
    end
else
    n_errors = [];
end

if ~isfield(classifier_pre, 'const') % When there is only one unique label in the node
    classifier.W = [];
else
    classifier.W = [classifier_pre(1,2).const classifier_pre(1,2).linear'];
end

classifier.ClassLabel = unique(labels_train)';
% classifier.labels_test = labels_test;

end

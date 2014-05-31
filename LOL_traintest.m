function [labels_pred, n_errors, classifier, labels_prob] = LOL_traintest( data_train, labels_train, data_test, labels_test, Opts )

% LDA train and test

if nargin<5,                    Opts = [];          end;
if ~isfield(Opts,'Priors'),     Opts.Priors = [];   end;

task.GMRAClassifier = Opts.GMRAClassifier;
[labels_pred, Proj, P, boundary] = LOL_classify(data_test,data_train,labels_train, task);
for i = 1: size(labels_pred{1},1)
ne(i) = sum(labels_pred{1}(i,:) ~= labels_test');
end
[n_errors, mse_ks] = min(ne);
labels_pred = labels_pred{1}(mse_ks, :);
% classifier = 0;
classifier.W = boundary;
classifier.Proj = Proj;
classifier.ClassLabel = unique(labels_train)';
labels_prob = 0;
% % Build LDA classifier
% classifier                           = LDA_train(data_train', labels_train',Opts.Priors);
% 
% % Test LDA classifier
% [n_errors, labels_pred, labels_prob] = LDA_test( classifier, data_test, labels_test );

end

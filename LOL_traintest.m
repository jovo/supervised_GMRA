function [labels_pred, n_errors, classifier, labels_prob] = LOL_traintest( data_train, labels_train, data_test, labels_test, Opts )

% LDA train and test

if nargin<5,                    Opts = [];          end;
if ~isfield(Opts,'Priors'),     Opts.Priors = [];   end;

% 
% task = {};
% task.LOL_alg = Opts.LOL_alg;
% task.ntrain = size(data_train',2);
% task.ks=unique(floor(logspace(0,log10(task.ntrain),task.ntrain)));
% Nks=length(task.ks);
% Nks=length(task.ks);
       
% Train LOL classifier

% LOL_classify input: data: N by D, 1 by N
[labels_pred, Proj, P, boundary] = LOL_classify(data_test',data_train',labels_train', Opts.task);
% disp('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA');
labels_pred = labels_pred{1}';

% boundary
% boundary{1}
% boundary{2}


% boundary{1}.W;
% boundary
% if isempty(data_test)
% boundary
% boundary{1}
% boundary{1}.W
% if ~isempty(boundary)
    classifier.W = boundary{1}.W;
% else
%    classifier.W = [];
% end

    classifier.Proj = Proj;
    classifier.ClassLabel = unique(labels_train)';    
% end
% disp('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB');
n_errors = 0;
labels_prob = [];
% classifier.W = boundary;
% classifier.Proj = Proj;
% classifier.ClassLabel = unique(labels_train)';
% labels_prob = [];
% n_errors = sum(labels_pred ~= labels_test');
% disp('b')

end

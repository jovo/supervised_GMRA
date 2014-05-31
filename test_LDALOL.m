% Test LOL for supervised_GMRA with fisheriris data
clear all
close all
clc

dir = fileparts(mfilename('fullpath'));
cd(dir); 
addpath(genpath(pwd));

load fisheriris

% Data 
% meas: measurements (n = 150 x d = 4)
% species of iris: 'setosa', 'versicolor', 'virginica  (150 x 1)
% Note: some points are overlapping..

%     variable for TRAINING. 
for i = 1: numel(species)
    if strcmp(species{i}, 'setosa')
        labels(i) = 0;
    elseif strcmp(species{i}, 'versicolor')
        labels(i) = 1;
    else
        labels(i) = 2;
    end
end

N = 100; % 2 groups of iris
% meas: X: n by d
X = meas(1:N,:);
% labels: Y: n by 1
Y = labels(1:N)';
clear i labels meas species 

idx = randperm(N);
data = X(idx,:);
labels = Y(idx, :);
clear X Y

Ntrain = 30; Ntest = N - Ntrain;
data_train = data(1:Ntrain, :);
data_test = data(Ntrain+1:end, :);
labels_train = labels(1:Ntrain, :);
labels_test = labels(Ntrain+1:end, :);

clear idx data labels 
whos

% training = meas(training_idx, 3:4);
% sample = meas(test_idx, 3:4);
% group_test = speciesNum(test_idx);
% group_train = speciesNum(training_idx);
% 
% % Scatter for training_idx
% SL_test =  meas(test_idx,3);
% SW_test =  meas(test_idx,4);
% SL_train = meas(training_idx,3);
% SW_train = meas(training_idx,4);
% 

% data: n by d, labels: n by 1, yet input as d by n and n by 1.
Opts = [];
[labels_pred_LDA, n_errors_LDA, classifier_LDA, ~] = LDA_traintest( data_train', labels_train, data_test', labels_test, Opts );
Opts.GMRAClassifier = {'DENL'};
[labels_pred_GMRALOL, n_errors_GMRALOL, classifier_GMRALOL, ~] = LOL_traintest( data_train, labels_train, data_test, labels_test, Opts );
Opts.GMRAClassifier = {'NNNL'};
[labels_pred_GMRALDA, n_errors_GMRALDA, classifier_GMRALDA, ~] = LOL_traintest( data_train, labels_train, data_test, labels_test, Opts );

% classifier_LDA.W(1,:) - classifier_LDA.W(2,:) == classifier_GMRALDA.W{3}
 
 
% Let's test the LDA_train vs LOL_train


% 
% Input = [ 1 2 3 7 8 9; 12 11 10 15 16 17 ]';% n = 6 by d = 2
% Target = [1 1 1 2 2 2]'; % n = 6 by d = 1
% [classifier_trained] = LDA_train(Input,Target); % ouput a struct
% classifier_trained.W % 2 by 3
% classifier_trained.ClassLabel % 2 by 1
% 
% [CLASS,ERR,POSTERIOR,LOGP,coef] = classify(Input,Input,Target,'linear'); 

% coef.type
% coef.name1
% coef.name2
% coef.const
% coef.linear
% 
% A = [coef(1,2).const; coef(1,2).linear]

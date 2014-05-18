% Test_GMRAClassifier.m is a matlab script that runs/compares GMRA_Classifier on
% sample data sets with different classifiers for testing/debugging purpose.

clear all
close all
clc

%% Pick a data set

pDataSetNames  = {'MNIST_Digits_70000', 'MNIST_Digits_1600', 'Gaussian_2', 'FisherIris' };
    
fprintf('\n Data Sets:\n');
for k = 1:length(pDataSetNames),
    fprintf('\n [%d] %s',k,pDataSetNames{k});
end;
fprintf('\n\n  ');

pDataSetIdx = input('Pick a data set to test: \n');

%% Load the data set

[X, TrainGroup, Labels] = LoadData(pDataSetNames{pDataSetIdx});

%% Pick a classifier 

pClassifierNames  = {'LDA', 'matlab_LDA', 'QDA', 'LOL'};

fprintf('\n Classifiers:\n');
for k = 1:length(pClassifierNames),
    fprintf('\n [%d] %s',k,pClassifierNames{k});
end;
fprintf('\n\n  ');

pClassifierIdx = input('Pick a classifier to test with: \n');
n_classifiers = numel(pClassifierIdx);

%% Add algorithm directory to search path

% Add GMRA directory
addpath(genpath('C:\Users\Billy\Desktop\GMRA\GMRA\'))
addpath(genpath('C:\Users\Billy\Desktop\GMRA\DiffusionGeometry\'))

% Add Classifier directory
for k = 1: n_classifiers
    [classifier_train{k}, classifier_test{k}] = AccessAlg(pClassifierNames{pClassifierIdx(k)});
end

for k = 1: n_classifiers
    if k > 1
       Opts.debugMRA = MRA{1}.debugMRA; % to use the same MRA for both classifiers.
    end
   Opts.Classifier = classifier_train{k};
   MRA{k} = GMRA_Classifier( X, TrainGroup, Labels, Opts);
   ClassifierResults{k} = GMRA_Classifier_test( MRA{k}, X, TrainGroup, Labels, classifier_test{k});
end

return;
 

%
% Opts = [];
% MRA_lda = GMRA_Classifier( X, TrainGroup, Labels, Opts);
% 
% % X: D by N, TrainGroup: 1 by N, Labels: 1 by N.
% Opts.Classifier = @matlabLDA_traintest;
% MRA_matlablda = GMRA_Classifier( X, TrainGroup, Labels, Opts);
% 
% % MRA_lda = GMRA_Classifier( X, TrainGroup, Labels);
% % addpath(genpath('C:\Users\Billy\Documents\GitHub\supervised_GRMA'))
% Opts = [];
% Opts.debugMRA = MRA_matlablda.debugMRA;
% MRA_lda2 = GMRA_Classifier( X, TrainGroup, Labels, Opts);
% 
% % Classifier_Test
% 
% ClassifierResults_lda2 = GMRA_Classifier_test( MRA_lda2, X, TrainGroup, Labels);
% ClassifierResults_lda3 = GMRA_Classifier_test( MRA_lda2, X, TrainGroup, Labels, @QDA_test);
% ClassifierResults_lda4 = GMRA_Classifier_test( MRA_lda2, X, TrainGroup, Labels, @LDA_test);
% ClassifierResults_matlablda = GMRA_Classifier_test( MRA_matlablda, X, TrainGroup, Labels, @matlabLDA_test); 
% 

%% Compare the Performance

fprintf('Comparison of errors: \n'); 
find(ClassifierResults_lda2.Test.errors ~= ClassifierResults_lda3.Test.errors)
find(ClassifierResults_lda2.Test.errors ~= ClassifierResults_lda4.Test.errors)
find(ClassifierResults_lda2.Test.errors ~= ClassifierResults_matlablda.Test.errors)

fprintf('Comparison of labels_node_pred: \n');
find(ClassifierResults_lda2.Test.Labels_node_pred{end} ~= ClassifierResults_lda3.Test.Labels_node_pred{end})
find(ClassifierResults_lda2.Test.Labels_node_pred{end} ~= ClassifierResults_lda4.Test.Labels_node_pred{end})
% find(ClassifierResults_lda2.Test.Labels_node_pred{end} ~= ClassifierResults_matlablda.Test.Labels_node_pred{end})

fprintf('Comparison of labels_node_prob: \n');
find(ClassifierResults_lda2.Test.Labels_node_prob{end} ~= ClassifierResults_lda3.Test.Labels_node_prob{end})
find(ClassifierResults_lda2.Test.Labels_node_prob{end} ~= ClassifierResults_lda4.Test.Labels_node_prob{end})
find(ClassifierResults_lda2.Test.Labels_node_prob{end} ~= ClassifierResults_matlablda.Test.Labels_node_prob{end})

fprintf('Comparison of Labels: \n');
find(ClassifierResults_lda2.Test.Labels ~= ClassifierResults_lda3.Test.Labels)
find(ClassifierResults_lda2.Test.Labels ~= ClassifierResults_lda4.Test.Labels)
find(ClassifierResults_lda2.Test.Labels ~= ClassifierResults_matlablda.Test.Labels)

find(ClassifierResults_lda2.Test.LabelsProb ~= ClassifierResults_lda3.Test.LabelsProb)
find(ClassifierResults_lda2.Test.LabelsProb ~= ClassifierResults_lda4.Test.LabelsProb)
find(ClassifierResults_lda2.Test.LabelsProb ~= ClassifierResults_matlablda.Test.LabelsProb)





function [ACC_QOQ, timing_QOQ] = GMRAClassifier_MNIST(NofPts_train, NofPts_test)
% QDA_MNIST runs QDA on the MNIST data for a binary classification problem

% Add the data directory to search path
addpath(genpath('/home/collabor/yb8/data/mnist/derived'));

% Add the algorithm directory to search path
addpath(genpath('/home/collabor/yb8/supervised_GMRA'));

% Load MNIST data
dummy = load('img_test.mat');
Xtest = dummy.images_test;
dummy = load('img_train.mat');
Xtrain = dummy.images_train;
dummy = load('labels_test.mat');
Ytest = dummy.labels_test;
dummy = load('labels_train.mat');
Ytrain = dummy.labels_train;
clear dummy

whos Xtest Xtrain Ytest Ytrain
% Sampling with the input: Nopts
if nargin == 2
    % Use randperm if you want to change the idx every trials
    Xtrain = Xtrain(1:NofPts_train);
    Ytrain = Ytrain(1:NofPts_train);
    Xtest = Xtest(1:NofPts_test);
    Ytest = Ytest(1:NofPts_test);
    % If you want to sample specific number of each digits, use:
    % [X, vLabels]=Generate_MNIST(100.*ones(1,10), struct('Sampling', 'RandN', 'QueryDigits', 0:9, 'ReturnForm', 'vector')); % Xtrain: n = 1000 x p = 784/ Ytrain: n = 1000 x p = 1
    % [Xtrain, Ytrain]=Generate_MNIST(50.*ones(1,10), struct('Sampling', 'RandN', 'QueryDigits', 0:9, 'ReturnForm', 'vector')); % Xtrain: n = 500 x p = 784/ Ytrain: n = 500 x p = 1
end

% Convert Y to binary classification problem
Ytrain(Ytrain<5) = 0; Ytrain(Ytrain>=5) = 1;
Ytest(Ytest<5) = 0; Ytest(Ytest>=5) = 1;
Ytrain(1:10)

% Convert the data to fit the input format for GMRA_Classifier
X = [Xtrain Xtest];
TrainGroup = [ones(1, size(Xtrain,2)) ones(1, size(Xtest,2))];
Labels = [Ytrain Ytest];
whos X TrainGroup Labels

% Run classifier
tic;

Opts = [];
MRA_lda = GMRA_Classifier( X, TrainGroup, Labels, Opts);

Opts.Classifier = @matlabLDA_traintest;
MRA_matlablda = GMRA_Classifier( X, TrainGroup, Labels, Opts);

% ClassifierResults_lda1 = GMRA_Classifier_test( MRA_lda, X, TrainGroup, Labels);
% ClassifierResults_lda2 = GMRA_Classifier_test( MRA_lda, X, TrainGroup, Labels, @QDA_test);
ClassifierResults_lda = GMRA_Classifier_test( MRA_lda, X, TrainGroup, Labels, @LDA_test);
ClassifierResults_matlablda = GMRA_Classifier_test( MRA_matlablda, X, TrainGroup, Labels, @matlabLDA_test); 

timing_GMRA = toc;

% Comparison of Performance
Yhat_GMRA_lda = ClassifierResults_lda.Test.Labels;
Yhat_GMRA_matlablda = ClassifierResults_matlablda.Test.Labels;

ACC_GMRA_lda = 1 - sum(Ytest~=Yhat_GMRA_lda')/(sum(Ytest==Yhat_GMRA_lda') + sum(Ytest~=Yhat_GMRA_lda'))
ACC_GMRA_matlablda = 1 - sum(Ytest~=Yhat_GMRA_matlablda')/(sum(Ytest==Yhat_GMRA_matlablda') + sum(Ytest~=Yhat_GMRA_matlablda'))

end

% % clear all;
% % 
% % %% Genearte the fisher's iris data
% % load('fisheriris');
% % 
% % speciesNum = nan(size(species));
% % for i = 1: numel(species)
% %     if strcmp(species{i}, 'setosa')
% %         speciesNum(i) = 0;
% %     elseif strcmp(species{i}, 'versicolor')
% %         speciesNum(i) = 1;
% %     else
% %         speciesNum(i) = 2;
% %     end
% % end
% % 
% % % N = 150; % 3 groups of iris
% % N = 100; % 2 groups of iris
% % idx = randperm(N);
% % ids = 50;
% % % test_idx = idx(1:ids);
% % training_idx = idx(ids+1: end);
% % 
% % %% Fitting the parameters for the GMRA_Classifier function.
% % %   X           : D by N matrix of N points in D dimensions
% % X = meas(1:N,:)';
% % %   TrainGroup  : row N vector, with 1's corresponding to columns of X to be used as training set.
% % TrainGroup = zeros(1,N);
% % TrainGroup(training_idx) = 1;
% % %   Labels      : row N vector of labels for the points
% % Labels = speciesNum(1:N)';
% % 
% % figure; 
% % % gscatter(X(3,TrainGroup == 1), X(4,TrainGroup == 1), Labels)
% % gscatter(X(3,:), X(4,:), Labels)
% % 
% % %         [Opts]      : structure with options
% % % Let's use the default options for now.
% % 
% % %% Add GMRA and GMRA_Classifier to the search path
% % 
% addpath('C:\Users\Billy\Desktop\GMRA')
% addpath(genpath('C:\Users\Billy\Desktop\GMRA\GMRA\'))
% addpath(genpath('C:\Users\Billy\Desktop\GMRA\Data_BMI\'))
% addpath(genpath('C:\Users\Billy\Desktop\GMRA\DiffusionGeometry\'))
% addpath(genpath('C:\Users\Billy\Desktop\GMRA\SyntheticData\'))
% addpath(genpath('C:\Users\Billy\Desktop\GMRA\DataSets'))
% % addpath(genpath('C:\Users\Billy\Desktop\Gallant'))
% % addpath(genpath('C:\Users\Billy\Desktop\Data'))
% % addpath(genpath('C:\Users\Billy\Desktop\Favorite\code_matlab\'))
% addpath(genpath('C:\Users\Billy\Documents\GitHub\supervised_GRMA'))
% % 
% % 
% % 
% % %% Run baby, run.
% % MRA = GMRA_Classifier( X, TrainGroup, Labels)
% % 
% % %% Fitting the parameters for the GMRA_Classifier_test function.
% % 
% % ClassifierResults = GMRA_Classifier_test( MRA, X, TrainGroup, Labels);
% % 
% % %% Results
% % ClassifierResults.Test.Labels
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Another Data Set: Sample from MNIST
clear all;
%% Add GMRA and GMRA_Classifier to the search path

% addpath('C:\Users\Billy\Desktop\GMRA')
addpath(genpath('C:\Users\Billy\Desktop\GMRA\GMRA\'))
% addpath(genpath('C:\Users\Billy\Desktop\GMRA\Data_BMI\'))
addpath(genpath('C:\Users\Billy\Desktop\GMRA\DiffusionGeometry\'))
% addpath(genpath('C:\Users\Billy\Desktop\GMRA\SyntheticData\'))
% addpath(genpath('C:\Users\Billy\Desktop\GMRA\DataSets'))
% addpath(genpath('C:\Users\Billy\Desktop\Gallant'))
addpath(genpath('C:\Users\Billy\Desktop\Data'))
% addpath(genpath('C:\Users\Billy\Desktop\Favorite\code_matlab\'))
addpath(genpath('C:\Users\Billy\Documents\GitHub\supervised_GRMA'))


% [X, vLabels]=Generate_MNIST([3, 3, 3], struct('Sampling', 'RandN', 'QueryDigits', [1, 6, 9], 'ReturnForm', 'vector')); % n = 9 x p = 784 (=28^2)
[X, vLabels]=Generate_MNIST([800, 800], struct('Sampling', 'RandN', 'QueryDigits', [0, 1], 'ReturnForm', 'vector')); % n = 9 x p = 784 (=28^2)
[N, D] = size(X);

%% Fitting the parameters for the GMRA_Classifier function.
%   X           : D by N matrix of N points in D dimensions
X = X';

%   Labels      : row N vector of labels for the points
Labels = vLabels';
clear vLabels

% Mix it up (a little bit) to test the classifier result
temp = X(:,801:1000);
X(:,801:1000) = X(:, 201:400);
X(:,201:400) = temp;

temp = X(:,1201:1400);
X(:,1201:1400) = X(:, 601:800);
X(:,601:800) = temp;

temp = Labels(:,801:1000);
Labels(:,801:1000) = Labels(:, 201:400);
Labels(:,201:400) = temp;

temp = Labels(:,1201:1400);
Labels(:,1201:1400) = Labels(:, 601:800);
Labels(:,601:800) = temp;

% Swap again for test data, just to check easily..

temp = X(:,1001:1100);
X(:,1001:1100) = X(:, 901:1000);
X(:,901:1000) = temp;

temp = Labels(:,1001:1100);
Labels(:,1001:1100) = Labels(:, 901:1000);
Labels(:,901:1000) = temp;

temp = X(:,1401:1500);
X(:,1401:1500) = X(:, 1301:1400);
X(:,1301:1400) = temp;

temp = Labels(:,1401:1500);
Labels(:,1401:1500) = Labels(:, 1301:1400);
Labels(:,1301:1400) = temp;


figure; imagesc(Labels); colorbar;

%   TrainGroup  : row N vector, with 1's corresponding to columns of X to be used as training set.
TrainGroup = zeros(1,N);
% training_idx = 1:N.*3/4;
training_idx = 1:N.*1/2;
TrainGroup(training_idx) = 1;

% A = X(403,:);
% B= X(404,:);
% figure; plot(A', B', 'x');
 
% figure; 
% gscatter(X(403,:), X(404,:), Labels)

%% Try LDA first with matlab's classify.m and mauro's LDA_traintest.m
% training = X(:,TrainGroup == 1)';
% sample = X(:,TrainGroup == 0)';
% group_train = Labels(:, TrainGroup == 1)';
% % training = X(:,1:400)';
% % sample = X(:,401:600)';
% % group_train = Labels(:, 1:400)';
% 
% [labels_pred_matlab, n_errors_matlab, labels_prob_matlab, ~, classifier_matlab] = classify(sample, training, group_train, 'linear', 'empirical');
% % sample: N by d, training: N by d, group_train: d by N
% [labels_pred, n_errors, classifier, labels_prob] = LDA_traintest( training', group_train, sample', []);
% 
% find(labels_pred_matlab ~= labels_pred)
% find(labels_prob_matlab(:,2) ~= labels_prob)

Opts = [];
MRA_lda = GMRA_Classifier( X, TrainGroup, Labels, Opts);
% X: D by N, TrainGroup: 1 by N, Labels: 1 by N.
Opts.Classifier = @matlabLDA_traintest;
MRA_matlablda = GMRA_Classifier( X, TrainGroup, Labels, Opts);

% MRA_lda = GMRA_Classifier( X, TrainGroup, Labels);
% addpath(genpath('C:\Users\Billy\Documents\GitHub\supervised_GRMA'))
Opts = [];
Opts.debugMRA = MRA_matlablda.debugMRA;
MRA_lda2 = GMRA_Classifier( X, TrainGroup, Labels, Opts);

% Classifier_Test

ClassifierResults_lda2 = GMRA_Classifier_test( MRA_lda2, X, TrainGroup, Labels);
ClassifierResults_lda3 = GMRA_Classifier_test( MRA_lda2, X, TrainGroup, Labels, @QDA_test);
ClassifierResults_lda4 = GMRA_Classifier_test( MRA_lda2, X, TrainGroup, Labels, @LDA_test);
ClassifierResults_matlablda = GMRA_Classifier_test( MRA_matlablda, X, TrainGroup, Labels, @matlabLDA_test); 

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




% %% Test data generation
% % clear all
% 
% N = 500;
% 
% p = 100; % n = 8;
% mu0 = -3; mu1 = 3;
% sigma0_d = 15; sigma0_c = 0.1;
% sigma1_d = 15; sigma1_c = 0.1;
% % 
% % p = 5; % n = 8;
% % mu0 = -3; mu1 = 3;
% % sigma0_d = 0.5; sigma0_c = 0.1;
% % sigma1_d = 0.5; sigma1_c = 0.1;
% 
% Mu0 = mu0.*rand(p,1);
% Mu1 = mu1.*rand(p,1);
% 
% % Generate two symmetric covariance matrices for two classes
% c = sigma0_c.*rand(p,1);
% Sigma0 = toeplitz(c) + sigma0_d.*eye(p); clear c;
% c = sigma1_c.*rand(p,1);
% Sigma1 = toeplitz(c) + sigma1_d.*eye(p); clear c;
% 
% X0 = mvnrnd(Mu0, Sigma0, N)'; % p = 2 x n = 500 matrix
% X1 = mvnrnd(Mu1, Sigma1, N)'; % p = 2 x n = 500 matrix
% X = [X0 X1]; % p = 2 x n = 1000
% Labels =[zeros(N,1); ones(N,1)];
% % N = 100; % 2 groups of iris
% N = 2.*N;
% %   TrainGroup  : row N vector, with 1's corresponding to columns of X to be used as training set.
% TrainGroup = zeros(1,N);
% % training_idx = 1:N.*3/4;
% training_idx = 1:N.*1/2;
% TrainGroup(training_idx) = 1;
% 
% % 
% % idx = randperm(2*N);
% % ids = 200;
% % test_idx = idx(1:ids);
% % training_idx = idx(ids+1: end);
% % 
% % training = X(:, training_idx)';
% % sample = X(:, test_idx)';
% % group_test = Labels(test_idx);
% % group_train = Labels(training_idx);
% 
% 
% 
% 
% 
% %% Run baby, run.
% MRA_lda = GMRA_Classifier( X, TrainGroup, Labels);
% ClassifierResults_lda = GMRA_Classifier_test( MRA_lda, X, TrainGroup, Labels);
% 
% %% Results
% % ClassifierResults_lda.Test.Labels
% % figure; imagesc(ClassifierResults_lda.Test.Labels);
% % 
% %% Let's try it with matlab's lda
% 
% Opts.Classifier = @matlabLDA_traintest;
% MRA_matlablda = GMRA_Classifier( X, TrainGroup, Labels, Opts);
% ClassifierResults_matlablda = GMRA_Classifier_test( MRA_matlablda, X, TrainGroup, Labels);
% 
% %%
% % addpath(genpath('C:\Users\Billy\Documents\GitHub\supervised_GRMA'))
% MRA_lda2 = GMRA_Classifier( X, TrainGroup, Labels);
% ClassifierResults_lda2 = GMRA_Classifier_test( MRA_lda2, X, TrainGroup, Labels);
% 
% 
% %% Results
% % ClassifierResults_matlablda.Test.Labels
% % figure; imagesc(ClassifierResults_matlablda.Test.Labels);
% 
% %% wtf, let's try it with jovo's lda
% % 
% % addpath(genpath('C:\Users\Billy\Documents\GitHub\supervised_GRMA'))
% % Opts.Classifier = @LDA_train_and_predict;
% % 
% % [Yhat, eta, parms] = LDA_train_and_predict(Xtrain, Ytrain, Xtest)
% % MRA_jovolda = GMRA_Classifier( X, TrainGroup, Labels, Opts)
% % ClassifierResults_jovolda = GMRA_Classifier_test( MRA_jovolda, X, TrainGroup, Labels);
% 
% 
% %% Results
% % ClassifierResults_matlablda.Test.Labels
% % figure; imagesc(ClassifierResults_matlablda.Test.Labels);
% 
% 
% 

% Test_GMRAClassifier.m is a matlab script that runs/compares GMRA_Classifier on
% sample data sets with different classifiers for testing/debugging purpose.

clear all
close all
clc

dir = fileparts(mfilename('fullpath'));
cd(dir); cd ..
addpath(genpath(pwd));

%% Pick a data set

pDataSetNames  = {'MNIST_HardBinary_T60K_t10K', 'MNIST_HardBinary_T5.0K_t5.0K',  'MNIST_HardBinary_T2.5K_t2.5K', 'MNIST_EasyBinary_T2.5K_t2.5K', 'MNIST_EasyBinary_T0.8K_t0.8K', 'MNIST_EasyBinary_T0.7K_t0.7K', 'MNIST_EasyBinary_T0.6K_t0.6K', 'MNIST_EasyBinary_T0.5K_t0.5K', 'MNIST_EasyBinary_T0.4K_t0.4K', 'MNIST_EasyBinary_T0.3K_t0.3K', 'MNIST_EasyBinary_T0.2K_t0.2K', 'MNIST_EasyTriple_T0.6K_t0.6K', 'MNIST_EasyTriple_T0.3K_t0.3K', 'MNIST_EasyBinary_T10_t10', 'Gaussian_2', 'FisherIris' };
    
fprintf('\n Data Sets:\n');
for k = 1:length(pDataSetNames),
    fprintf('\n [%d] %s',k,pDataSetNames{k});
end;
fprintf('\n\n  ');

pDataSetIdx = input('Pick a data set to test: \n');

%% Load the data set

[X, TrainGroup, Labels] = LoadData(pDataSetNames{pDataSetIdx});

%% Pick a classifier 

pClassifierNames  = {'LDA', 'matlab_LDA', 'QDA', 'LOL: LDA', 'LOL: LOL'};

fprintf('\n Classifiers:\n');
for k = 1:length(pClassifierNames),
    fprintf('\n [%d] %s',k,pClassifierNames{k});
end;
fprintf('\n\n  ');

pClassifierIdx = input('Pick a classifier to test with: \n');
n_classifiers = numel(pClassifierIdx);

%% Add algorithm directory to search path

% Add GMRA directory
addpath(genpath('/home/collabor/yb8/GMRA/'))
addpath(genpath('/home/collabor/yb8/DiffusionGeometry/'))

% Add Classifier directory
for k = 1: n_classifiers
    [classifier_train{k}, classifier_test{k}, LOL_alg{k}] = AccessAlg(pClassifierNames{pClassifierIdx(k)});
end

for k = 1: n_classifiers
	if k > 1
		Opts.debugMRA = MRA{1}.debugMRA; % to use the same MRA for both classifiers.
   	end
   		Opts.Classifier = classifier_train{k};
   		Opts.LOL_alg = LOL_alg{k};
    	if exist('MRAsaved.mat') == 0
		disp('No previous MRA is saved. Running GMRA_Classifier.')
        	MRA{k} = GMRA_Classifier( X, TrainGroup, Labels, Opts);
	else	
		MRA{1}= load('MRAsave.mat');
	end
%   MRA{1}
%   MRA{1}.cp
%   MRA{1}.Classifier.Classifier
%   isempty(MRA{1}.Classifier.Classifier)
%    MRA{1}.Classifier.Classifier{end}{1}.W{1}
	disp('# of the nodes from the resulted MRA: ');
	length(MRA{1}.cp)
	disp('# of the active nodes from the resulted MRA: ');
	length(MRA{1}.Classifier.activenode_idxs)
  	Opts = [];
   	Opts.LOL_alg = LOL_alg{k};
%	disp('checking the classifier input to GMRA_Classifier_test')
%	MRA{1}.Classifier.Classifier
	disp('checking the min_ks transfer.........................................................')
	MRA{1}.min_ks
	ClassifierResults{k} = GMRA_Classifier_test( MRA{k}, X, TrainGroup, Labels, classifier_test{k}, Opts);
end


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

%% Accuracy

labels_test = Labels(TrainGroup == 0);
count = 0;		
for k = 1: n_classifiers
	if isequal(classifier_test{k}, @LOL_test)
		for i = 1: length(MRA{1}.Classifier.activenode_idxs)
			thisnode = ClassifierResults{1}.Test.Labels.labels_pred{i};
			node_idx = ClassifierResults{1}.Test.Labels.idx{i};
			node_true = labels_test(node_idx);
			for j = 1: length(thisnode)
				disp('j: ')
				j		
				size(node_true')
				size(thisnode{j})
				if isempty(node_true)
					node_error(j) = 0;
				else
					node_error(j) = sum(node_true' ~= thisnode{j});
				end
			end
			min_node_error(i) = min(node_error);
			count = count + numel(node_true);
		end
 	   	ACC_GMRAClassifier(k) = 1 - sum(min_node_error)./numel(Labels(TrainGroup == 0))
	else
	   	ACC_GMRAClassifier(k) = 1 - numel(find(ClassifierResults{k}.Test.Labels ~= Labels(TrainGroup == 0)))./numel(Labels(TrainGroup == 0))
	end
	disp('the count: ')
	count
end


% save('ACC_GMRAClassifier', 'ACC_GMRAClassifier');
return;
%% Accuracy of Simple-LDA for comparison
X = X'; Labels = Labels'; % D by N to N by D
sample = X(TrainGroup == 0, :);
training = X(TrainGroup == 1, :);
group = Labels(TrainGroup == 1, :);
test = Labels(TrainGroup == 0, :);

% Mauro's LDA
[Labels_mauroLDA, ~, ~, ~] = LDA_traintest( training', group', sample', test', Opts);
ACC_MauroLDA  = 1 - numel(find(Labels_mauroLDA ~= test))./numel(test)
save('ACC_MauroLDA', 'ACC_MauroLDA');

% Matlab's LDA
% class = classify(sample, training, group);
% ACC_LDA  = 1 - numel(find(class ~= test))./numel(test)
% save('ACC_LDA', 'ACC_LDA');

%% Compare the Performance

% fprintf('Comparison of errors: \n'); 
% find(ClassifierResults_lda2.Test.errors ~= ClassifierResults_lda3.Test.errors)
% find(ClassifierResults_lda2.Test.errors ~= ClassifierResults_lda4.Test.errors)
% find(ClassifierResults_lda2.Test.errors ~= ClassifierResults_matlablda.Test.errors)

% fprintf('Comparison of labels_node_pred: \n');
% find(ClassifierResults_lda2.Test.Labels_node_pred{end} ~= ClassifierResults_lda3.Test.Labels_node_pred{end})
% find(ClassifierResults_lda2.Test.Labels_node_pred{end} ~= ClassifierResults_lda4.Test.Labels_node_pred{end})
% find(ClassifierResults_lda2.Test.Labels_node_pred{end} ~= ClassifierResults_matlablda.Test.Labels_node_pred{end})

% fprintf('Comparison of labels_node_prob: \n');
% find(ClassifierResults_lda2.Test.Labels_node_prob{end} ~= ClassifierResults_lda3.Test.Labels_node_prob{end})
% find(ClassifierResults_lda2.Test.Labels_node_prob{end} ~= ClassifierResults_lda4.Test.Labels_node_prob{end})
% find(ClassifierResults_lda2.Test.Labels_node_prob{end} ~= ClassifierResults_matlablda.Test.Labels_node_prob{end})

% fprintf('Comparison of Labels: \n');
% find(ClassifierResults_lda2.Test.Labels ~= ClassifierResults_lda3.Test.Labels)
% find(ClassifierResults_lda2.Test.Labels ~= ClassifierResults_lda4.Test.Labels)
% find(ClassifierResults_lda2.Test.Labels ~= ClassifierResults_matlablda.Test.Labels)

% find(ClassifierResults_lda2.Test.LabelsProb ~= ClassifierResults_lda3.Test.LabelsProb)
% find(ClassifierResults_lda2.Test.LabelsProb ~= ClassifierResults_lda4.Test.LabelsProb)
% find(ClassifierResults_lda2.Test.LabelsProb ~= ClassifierResults_matlablda.Test.LabelsProb)





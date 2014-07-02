function [classifier_train, classifier_test, LOL_alg] = AccessAlg(pClassifierName)
% LoadData loads the data and the labels for the classification.
% X: D by N (= Ntrain + Ntest) 
% TrainGroup: 1 by N with 1's corresponding to columns of X to be used as training set.
% Labels: 1 by N of labels

switch pClassifierName
    
    case 'LDA'
        classifier_train = @LDA_traintest;
        classifier_test = @LDA_test;
        LOL_alg = [];
        
    case 'matlab_LDA'
        classifier_train = @matlabLDA_traintest;
        classifier_test = @matlabLDA_test;
        LOL_alg = [];

    case 'QDA'
        classifier_train = @matlabLDA_traintest;
        classifier_test = @matlabLDA_test;
        LOL_alg = [];
        
    case 'LOL: LDA'
        classifier_train = @LOL_traintest;
        classifier_test = @LOL_test;
        LOL_alg = {'NNNL'};
        
%         task = {};
%         task.LOL_alg = Opts.LOL_alg;
%         task.ntrain = size(data_train',2);
%         task.ks=unique(floor(logspace(0,log10(task.ntrain),task.ntrain)));
%         Nks=length(task.ks);
        
    case 'LOL: LOL'
        classifier_train = @LOL_traintest;
        classifier_test = @LOL_test;
        LOL_alg = {'DENL'};

    case 'LOL: embeddingLOL'
	classifier_train = @LDA_traintest;
        classifier_test = @LDA_test;
	LOL_alg = {'DENL'};
end


function [classifier_train, classifier_test] = AccessAlg(pClassifierName)
% LoadData loads the data and the labels for the classification.
% X: D by N (= Ntrain + Ntest) 
% TrainGroup: 1 by N with 1's corresponding to columns of X to be used as training set.
% Labels: 1 by N of labels

switch pClassifierName
    
    case 'LDA'
        addpath(genpath('/home/collabor/yb8/supervised_GRMA/'))
        classifier_train = @LDA_traintest;
        classifier_test = @LDA_test;
        
    case 'matlab_LDA'
        addpath(genpath('/home/collabor/yb8/supervised_GRMA/'));
        classifier_train = @matlabLDA_traintest;
        classifier_test = @matlabLDA_test;

    case 'QDA'
        addpath(genpath('/home/collabor/yb8/supervised_GRMA/'))
        
    case 'LOL'
        addpath(genpath('/home/collabor/yb8/supervised_GRMA/'))

end


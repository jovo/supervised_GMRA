% This test is considered a "true" holdout because some points are held
% back even from the original GWT. An estimate is made for the optimal
% scales to use and the error rates using the original cross-validation on the
% training set, and then the truly held out (test) points are tested on the model
% (which was built using all of the training points)

% clear all
% close all
% clc

stream0 = RandStream('mt19937ar','Seed',2);
RandStream.setGlobalStream(stream0);

%% Go parallel
% if matlabpool('size')==0,
%     matlabpool('OPEN',6);
% end;

%% Pick a data set
pExampleNames  = {'MNIST_Digits_Full', 'MNIST_Digits_Subset','YaleB_Faces','croppedYaleB_Faces','ScienceNews', ...
                  'ScienceNewsDefaultParams', ...
                  'Medical12images','Medical12Sift','CorelImages','CorelSift', ...
                  'Olivetti_faces', ...
                  '20NewsAllTrain', '20NewsAllTest', '20NewsAllCombo', ...
                  '20NewsSubset1','20NewsSubset2tf','20NewsSubset3','20NewsSubset4', ...
                  '20NewsCompSetOf5'};

fprintf('\n Examples:\n');
for k = 1:length(pExampleNames),
    fprintf('\n [%d] %s',k,pExampleNames{k});
end;
fprintf('\n\n  ');

pExampleIdx = input('Pick an example to run: \n');

pGWTversion = 0;

% Generate data and choose parameters for GWT
[X, GWTopts, imgOpts] = EMo_GenerateData_and_SetParameters(pExampleNames{pExampleIdx});
fprintf(1, '\n\n');
GWTopts.GWTversion = pGWTversion;

% Make an external copy that will copy from for holdout groups along with X
Labels = imgOpts.Labels;
imgOpts = rmfield(imgOpts, 'Labels');


%%  Construct the GMRA classifier
[GMRA_Classifier, GMRA] = GMRA_LDA( X, Labels, struct('GMRAopts',GWTopts) );
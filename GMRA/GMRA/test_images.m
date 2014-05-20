clear all;close all;clc;

global gW;

%% Parameters
pPatchSize = 16;

%% Load image
I=imread('Lena.jpg');
I=double(I);
I=I/max(max(I));

%% Downsample image
I = I(1:2:size(I,1),1:2:size(I,2));

%% Construct data set
gW.X = filtergraph(I,'ptch',pPatchSize); 


%% set GWT parameters
GWTopts = struct();
GWTopts.ManifoldDimension = 0; % if 0, then determine locally adaptive dimensions using the following fields:
GWTopts.errorType = 'relative'; % or absolute
GWTopts.threshold0 = 0.5; % threshold for choosing pca dimension at each nonleaf node
GWTopts.precision  = 1e-1; % only for leaf nodes
% The following thresholds are used in the code construct_GMRA.m
GWTopts.threshold1 = 1e-1; % threshold of singular values for determining the rank of each ( I - \Phi_{j,k} * \Phi_{j,k} ) * Phi_{j+1,k'}
GWTopts.threshold2 = 5e-2; % threshold for determining the rank of intersection of ( I - \Phi_{j,k} * \Phi_{j,k} ) * Phi_{j+1,k'}
% The following parameter .pruning determines which version of geometric wavelets to use
GWTopts.pruning = 1;
% whether to use best approximations
GWTopts.addTangentialCorrections = true;
% whether to sparsify the scaling functions and wavelet bases
GWTopts.sparsifying = false;
% whether to split the wavelet bases into a common intersection and
% children-specific parts
GWTopts.splitting = false;
% METIS parameters
GWTopts.knn = 50;
GWTopts.knnAutotune = 30;
GWTopts.smallestMetisNet = 20;


%% Construct Geometric Wavelets
fprintf('\n Constructing Geometric Wavelets...');
gW = GMRA(gW.X,GWTopts);
fprintf('done.');

%% Computing all wavelet coefficients 
fprintf('\n Computing GWT of original data...');
[gW,Data] = GWT_trainingData(gW, gW.X);
fprintf('done.');

fprintf('\n');

%% Display results
% Display the GWT coefficients
GWT_DisplayCoeffs( gW, Data );

% Plot GWTapproximation error
GWT_DisplayApproxErr( gW, Data );

% Plot some elements in the dictionary

return;
function [GWT, Data] = applyGWRA2ImagePatches(X, GWTopts)

%% set GWT parameters

if nargin<2, GWTopts = struct(); end

% whether to use best approximations
GWTopts.addTangentialCorrections = false;

% whether to sparsify the scaling functions and wavelet bases
GWTopts.sparsifying = true;

% whether to split the wavelet bases into a common intersection and
% children-specific parts
GWTopts.splitting = false;

% whether to output time
GWTopts.verbose = 1;

% threshold for wavelet coefficients
GWTopts.coeffs_threshold = 0; %GWTopts.precision/10;

% method for shrinking the wavelet coefficients
GWTopts.shrinkage = 'hard';

% whether to avoid using the scaling functions at the leaf nodes and
% instead using the union of their wavelet bases and the scaling functions
% at the parents
GWTopts.avoidLeafnodePhi = true;

% GWT parameters that need to be set separately
%GWTopts.ManifoldDimension = 4;
GWTopts.errorType = 'relative';
GWTopts.threshold0 = 0.75; % threshold for choosing pca dimension at each nonleaf node
GWTopts.precision  = 0.01; % only for leaf nodes

%% image parameters
imgOpts = struct();
imgOpts.imageData = true;
imgOpts.imR = sqrt(size(X,1));
imgOpts.imC = imgOpts.imR;
imgOpts.isCompressed = false;

%% construct geometric wavelets
GWTopts.GWTversion = 0;
GWT = GMRA(X, GWTopts);

%% Computing all wavelet coefficients 
[GWT, Data] = GWT_trainingData(GWT, X);

%% display results
GWT_displayResults(GWT,Data,imgOpts);


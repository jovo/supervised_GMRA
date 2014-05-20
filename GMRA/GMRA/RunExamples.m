%%%%%%%%%%%%%%%% script for running GMRA %%%%%%%%%%%%%%%%%%
clear all
close all
clc

%% Go parallel
if matlabpool('size')==0,
    matlabpool
end;

%% Pick a data set

global dx T dy

dx = [];

pExampleNames  = {'MNIST_Digits','YaleB_Faces','croppedYaleB_Faces','SpikeBins','NaturalImagePatches','ScienceNews', 'IntegralOperator','MeyerStaircase', ...
                           'SwissRoll', 'valley', 'S-Manifold','Oscillating2DWave','D-Ball','D-Sphere', 'D-Cube','D-FlatTorus','Cosine','Signal1D_1','D-Gaussian'};

fprintf('\n Examples:\n');
for k = 1:length(pExampleNames),
    fprintf('\n [%d] %s',k,pExampleNames{k});
end;
fprintf('\n\n  ');

pExampleIdx = input('Pick an example to run: \n');

%% Choose a GWT version
GWTversions  = {'Vanilla GWT','Orthogonal GWT','Pruning GWT'};
methodLabels = [0 1 2];
fprintf('\n Geometric Wavelets version:\n');
for k = 1:length(GWTversions),
    fprintf('\n [%d] %s',methodLabels(k),GWTversions{k});
end;
fprintf('\n\n  ');

pGWTversion = input('Pick a version of the GWT to run: \n');

%% Generate data and choose parameters for GWT
[X, GWTopts, imgOpts] = GenerateData_and_SetParameters(pExampleNames{pExampleIdx});
whos X
%% Construct geometric wavelets
GWTopts.GWTversion = pGWTversion;
GWT = GMRA(X, GWTopts);

%% Compute all wavelet coefficients 
[GWT, Data] = GWT_trainingData(GWT, X);

%% Display results
GWT_displayResults(GWT,Data,imgOpts);
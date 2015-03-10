function [gW, Data] = GWT_trainingData(gW, X)

% This code takes the GMRA structure and the training data as input
% and computes the wavelet coefficients of the points and the
% reconstructions of the data at all scales.

% Forward GWT
Data = FGWT_trainingData(gW, X);

% threshold coefficients
%Data.CelWavCoeffs(:,2:end) = threshold_coefficients(Data.CelWavCoeffs(:,2:end), gW.opts);

% simplify the tree if possible
%[gW, Data] = simplify_the_GWT_tree(gW, Data);

% Inverse GWT
[Data.Projections, Data.TangentialCorrections] = IGWT_trainingData(gW, Data.CelWavCoeffs);

return;


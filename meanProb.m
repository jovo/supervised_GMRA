function [ meanprob ] = meanProb( Cov )
%meanProb.m computes the log of mean likelihood of the multivariate normal distribution 
% Input: covariance matrix, D-k by D-k for conditional
% covariance matrix with k random variables with known values with ambient dimension = D.
% Output: the log mean likelihood of the (conditional) distribution

Dk = size(Cov,1); % equals to D-k
scaleTerm = -(Dk)*log(2*sqrt(pi));
%disp('max diag, min diag of cov')
% format long
%max(diag(Cov))
%min(diag(Cov))
% disp('diagTerm')
diagTerm = -1*sum(log(diag(Cov)));
logdetTerm = -0.5*logdet(Cov);
meanprob = scaleTerm + diagTerm + logdetTerm;

end


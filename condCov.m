function [ condcov ] = condCov( pooledCov, k )
%condCov computes the conditional covariance of D-k dimension depending on k random
%variables with known value
% Input
% pooledCov: the pooled covariance of the D dimensional multivariate normal
% distribution
% k        : the intrinsic (reduced) dimension 
% Output
% condcov  : the covariance of the conditional distribution of D-k
% dimensions depending on the k dimensions where the values are known.

CovXX = pooledCov(1:k, 1:k);
CovXY = pooledCov(1:k, k+1:end);
CovYX = pooledCov(k+1:end, 1:k);
CovYY = pooledCov(k+1:end, k+1:end);
%disp('diag of fullCovMat')
%max(diag(pooledCov))
%min(diag(pooledCov))

condcov = CovYY - CovYX*pinv(CovXX)*CovXY;
%disp('diag of condcov')
%max(diag(condcov))
%min(diag(condcov))

end


% Normal Mean Estimation for each classes:  mean for each classes
% Normal Covariance matrix Estimation:      pooled covariance of classes
% Prior Distribution for each classes:      (equal probability as in classify.m)
% BIC = -2*ln(max_L) + k*(ln(n) - ln(2*pi))
% Cov_mat: 

% For each k, I need to find the corresponding parameters as follows:
normalDist = -0.5*(log(det(cov_mat)) - (X-mean)'*inv(cov_mat)*(X-mean) - D*log(2*pi));
max_L = sum(normalDist*prior)
BIC 




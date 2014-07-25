function BIC = computeBIC(X_train, Labels_train, X_test, paramsBIC, classifier)
%computeBIC.m computes the BIC value of each LOL model with different k.

%% Parameters for BIC

disp('PriorProb, PooleCov, invCov, GroupMean, Group, nGroup')
   size(paramsBIC.PriorProb)
   size(paramsBIC.PooledCov)
   size(paramsBIC.invCov)
   size(paramsBIC.GroupMean)
   A = paramsBIC.PriorProb;
   B = paramsBIC.PooledCov;
   C = paramsBIC.invCov;
   D = paramsBIC.GroupMean;
%   E = paramsBIC.Group;
   F = paramsBIC.nGroup;
whos A B C D E F
%% Compute BIC

% Sort by Groups
disp('check size of classifier')
size(classifier)
size(classifier.W)
disp('check X_train size')
X_train = X_train';
size(X_train)

disp('check size of invCov')
size(paramsBIC.invCov)
% For each k, I need to find the corresponding parameters as follows:

logdetcov = -0.5*logdet(paramsBIC.PooledCov); 
disp('size of logdetcov, logdetcov')
size(logdetcov)
logdetcov

% What is paramsBIC.Group
disp('what is paramsBIC.Group')
size(paramsBIC.Group)
% paramsBIC.Group
%        Group      = (Target == ClassLabel(i));
for i = 1: numel(classifier.ClassLabel)
     Group = (Labels_train == ClassLabel(i));
     centeredX = bsxfun(@minus, X_train(Group,:), paramsBIC.GroupMean);

end  

disp('checking the size of X_train for this group and Group mean')
size(X_train(paramsBIC.Group,:))
size(paramsBIC.GroupMean)

% disp('size
% centeredX = bsxfun(@minus, X_train(paramsBIC.Group,:), paramsBIC.GroupMean);
disp('checking the size of the centeredX')
size(centeredX)
% centeredX =  X_train(paramsBIC.Group,:)-paramsBIC.GroupMean;
disp('hi')
% normalDist = -0.5*logdet(paramsBIC.PooledCov) - centeredX'*paramsBIC.invCov*centeredX - 
% normalDist = -0.5*logdet(paramsBIC.PooledCov) - (X_train(paramsBIC.Group,:)-paramsBIC.GroupMean)'*paramsBIC.invCov*


% normalDist = -0.5*(log(det(cov_mat)) - (X-mean)'*inv(cov_mat)*(X-mean) - D*log(2*pi));

% max_L = sum(normalDist*prior)
BIC = 0; 




end


function vApproxErr = GWT_ApproxError( cX, cProjections, cOpts )

%
% IN:
%   cX              : N by D matrix of N data points in R^D
%   cProjections    : Projections of data, in a N by D by J tensor, J being the number of scales
%   [cOpts]         : structure with the following fields:
%                       norm    : p-norm to be used to measure error at each scale, between cX and cProjections(:,:,j). 
%                                 Can be any p\in(0,+\infty]. Default: 2.
%                       relative: whether it is relative norm or absolute norm. Default: false.
%
% OUT:
%   vApproxErr      : J column vector of errors at different scales (fine to coarse). 
%                     It is normalized to be scale-invariant, by dividing by the variance of X, and to be independent of the number of points: 
%                     if relative == false
%                           (1/N \sum_{i=1}^N ||X(i,:)-X_j(i,:)||_2^p)^(1/p)
%                     if relative == true
%                           (1/N \sum_{i=1}^N (||X(i,:)-X_j(i,:)||_2/||X(i,:)||_2)^p)^(1/p)
%                     where X_j is the projection of X on the j-th scale.
%
% EXAMPLE:
%   vApproxErr=GWT_ApproxError(Data.X,Data.Projections,struct('norm','fro'));
%
% 
% (c) 2010 Mauro Maggioni and Guangliang Chen, Duke University
% Contact: {mauro, glchen}@math.duke.edu

if nargin<3,                    cOpts = [];                 end;
if ~isfield(cOpts,'norm'),      cOpts.norm = 2;             end;
if ~isfield(cOpts,'relative'),  cOpts.relative = false;     end;

J           = size(cProjections,3);
vApproxErr  = zeros(J,1);

if cOpts.relative,
    lNormXk = sum(cX.^2,1);
    lNormXk(lNormXk==0)=1;
end;

for j = 1:J,
    % compute L^2 approximation error
    lNorms = sum((cX-cProjections(:,:,j)).^2,1);
    
    if cOpts.relative
        lNorms  = lNorms./lNormXk;
    end
    
    switch cOpts.norm
        case 1
            vApproxErr(j) = mean(sqrt(lNorms));
        case 2
            vApproxErr(j) = sqrt(mean(lNorms));
        case inf
            vApproxErr(j) = sqrt(max(lNorms)); 
        otherwise
            vApproxErr(j) = (mean(lNorms.^(cOpts.norm/2)))^(1/cOpts.norm); 
    end;
    
end;

return
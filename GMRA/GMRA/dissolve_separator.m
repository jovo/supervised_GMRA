function IniLabels = dissolve_separator(X, cp, cmember, goodNets, method)

% This function assigns the points in the separators of the metis tree to the
% "nearest" leaf nodes. It uses one of the following three methods:
%  
% 'kmeans':  The centers of the leaf nodes are used as initial centroids
%              given to kmeans to obtain a new partition of the data  
% 'centers': The points in the separators are assinged to the leaf nodes
%              containing the closest center.
% 'ends':    The points in the separators are assigned to the leaf nodes
%              containing the closest point.
%
% INPUT
%   X       : the data set
%   cp      : tree structure as returned from nesdis
%   cmember : membership of the points in the original metis tree
%   goodNets: an initialization of reasonably good points
%   method  : one of the three methods above
%   
% OUTPUT 
%   IniLabels: labels of the data points w.r.t. the leaf nodes

if nargin<2
    method = 'centers';
end

%%
flags = zeros(1, max(cmember));
flags(cp(1:end-1)) = 1;

isInSeparator = false(1,size(X,1));
isInSeparator(flags(cmember)>0) = true;
% The above four lines are equivalent to the following:
% isInSeparator = ismember(cmember, gW.cp);

nGoodNets   = length(goodNets);

goodCenters = zeros(size(X,1),nGoodNets);
for i = 1:nGoodNets
    goodCenters(:,i) = mean(X(:,cmember == goodNets(i)),2);
end

switch lower(method)
    
    case 'kmeans'
        
        inds = kmeans(X', [], 'start', goodCenters');
        IniLabels = goodNets(inds);
        
    case 'centers'
        
        if sum(isInSeparator)<20000
            dists = repmat(sum(X(:,isInSeparator).^2,1), nGoodNets,1) + repmat((sum(goodCenters.^2,1)), sum(isInSeparator),1)' - 2*goodCenters'*X(:,isInSeparator);         % Fast but memory consuming
        else
            dists = distancessq( goodCenters, X(:,isInSeparator) );                                                                                                         % Slow but memory-parsimonious
        end;
        [~, inds] = min(dists, [], 1);
        IniLabels = cmember;
        IniLabels(isInSeparator) = goodNets(inds);
        
    case 'ends'
        
        lens = sum(X.^2, 1);
        goodPoints = ~isInSeparator;
        dists = repmat(lens(isInSeparator), 1, sum(goodPoints)) + repmat(lens(goodPoints)', sum(isInSeparator),1) - 2*X(:,isInSeparator)'*X(:,goodPoints);
        
        [~, inds] = min(dists, [], 2);
        
        IniLabels = cmember;
        cmem = cmember(goodPoints);
        IniLabels(isInSeparator) = cmem(inds);
        
end

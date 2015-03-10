function gW = trimming_small_leaves(gW,isaCompletedTree)

% This function trims the metis tree by removing the small leaf nodes
% whose parent is also within the given precision, when either the manifold
% dimension or a vector of absolute errors (with minimum at the leaf nodes)
% is used.

if (gW.opts.ManifoldDimension>0) %|| (strcmpi(gW.opts.errorType, 'absolute') && gW.opts.precision < min(gW.opts.threshold0(~gW.isaleaf)))
    
    if nargin==1,
        isaCompletedTree = false; % svd analysis was only performed at the leaf nodes
    end
    
    %%
    nAllNets  = numel(gW.cp); % number of nodes in the tree
    J           = max(gW.Scales); % number of scales
    flags      = ones(1, nAllNets);
    
    %%
    j = J-1;
    n = 2;
    while j>0 && n>0
        
        nodes = find(gW.Scales == j);
        n = sum(gW.isaleaf(nodes));
        nodes = nodes(~gW.isaleaf(nodes));
        
        for i = 1:numel(nodes)
            
            node = nodes(i);
            
            if ~isaCompletedTree % construct this node
                children = (gW.cp==node);
                gW.PointsInNet{node} = [gW.PointsInNet{children}];
                gW = local_SVD_analysis(gW,node);
            end
            
            Y = bsxfun(@minus,gW.X(:,gW.PointsInNet{node}),gW.Centers{node})';
            if strcmpi(gW.opts.errorType, 'absolute'),
                approxError = sqrt(mean(sum(Y.^2,2) - sum((Y*gW.ScalFuns{node}).^2,2)));
            else
                approxError = (norm(Y*gW.ScalFuns{node}, 'fro') / norm(Y, 'fro'))^2;
            end
            
            if approxError <= gW.opts.precision
                flags(get_offspring(gW.cp, node)) = -1;
                gW.IniLabels(gW.PointsInNet{node}) = node;
                n = n+1;
            end
            
        end
        
        j = j-1;
        
    end
    
    %%
    gW = get_subtree(gW,flags);
    
end

return;


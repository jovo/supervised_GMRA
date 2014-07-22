% testK with mindim.m
gMRA.Centers{node}  = mean(lPtsInNode,2);
Y                   = bsxfun(@minus,lPtsInNode,gMRA.Centers{node});         % Centered data in the current node
gMRA.Radii(node)    = sqrt(max(sum(Y.^2,1)));

%% Compute local SVD
if gMRA.isaleaf(node) || gMRA.opts.ManifoldDimension==0 
    %% Local dimension is not fixed, but based on local singular value decay
    [V,S,~]             = randPCA(Y,min([min(size(Y)),gMRA.opts.MaxDim]));                      % Use fast randomized PCA
    remEnergy           = sum(sum(Y.^2))-sum(diag(S).^2);
    gMRA.Sigmas{node}   = ([diag(S); sqrt(remEnergy)]) /sqrt(gMRA.Sizes(node));
    if gMRA.isaleaf(node) || gMRA.opts.pruning
        errorType = gMRA.opts.errorType;
    else
        errorType = 'relative';
    end
    reqDim = min(numel(diag(S)), mindim(gMRA.Sigmas{node}, errorType, gMRA.opts.threshold0(node))); 
    
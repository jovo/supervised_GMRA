function gMRA = local_SVD_analysis(gMRA,node)

% This function performs local svd analysis at the input node.
% In particular, it computes local mean and basis vectors, as well as the
% singular values.
   
PointsInThisNode    = gMRA.PointsInNet{node};
gMRA.Sizes(node)    = numel(PointsInThisNode);
lPtsInNode          = gMRA.X(:,PointsInThisNode);
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
    gMRA.ScalFuns{node} = V(:,1:reqDim);
else
    %% Manifold dimension is given
    [V,S,~]             = randPCA(Y,min([gMRA.opts.ManifoldDimension,min(size(Y))]));           % Use fast randomized PCA
    gMRA.Sigmas{node}   = diag(S)/sqrt(gMRA.Sizes(node));
    if size(V,2)<gMRA.opts.ManifoldDimension,
        V = [V,zeros(gMRA.opts.AmbientDimension,gMRA.opts.ManifoldDimension-size(V,2))];
    end;
    gMRA.ScalFuns{node} = V(:,1:gMRA.opts.ManifoldDimension);
end

%% Pruning
if gMRA.opts.pruning, % minimal encoding cost pruning
    gMRA.epsEncodingCosts(node) = (gMRA.Sizes(node)+gMRA.opts.AmbientDimension) * reqDim + gMRA.opts.AmbientDimension;
end

%% Sparsify the local dictionary if requested
if gMRA.opts.sparsifying,    
%     gMRA.Projections{node} = Y*gMRA.ScalFuns{node}*(gMRA.ScalFuns{node})';
    if gMRA.isaleaf(node) || (~gMRA.isaleaf(node) && gMRA.opts.addTangentialCorrections)
        gMRA.Projections{node} = gMRA.ScalFuns{node}*gMRA.ScalFuns{node}'*Y + repmat(gMRA.Centers{node}, 1,gMRA.Sizes(node));
    else %~isempty(children) && ~gMRA.opts.addTangentialCorrections
        children = (gMRA.cp==node);
        childrenProj = cat(2, gMRA.Projections{children});
        if ~isempty(childrenProj)                        
            gMRA.Projections{node} = gMRA.ScalFuns{node}*(gMRA.ScalFuns{node})'*(childrenProj-repmat(gMRA.Centers{node}, 1,gMRA.Sizes(node))) + repmat(gMRA.Centers{node}, 1,gMRA.Sizes(node));        
        end;
        %computeWaveletCoeffcients(cat(2, gMRA.Projections{children})-repmat(gMRA.Centers{node}, 1,gMRA.Sizes(node)), gMRA.ScalFuns{node}, gMRA.opts.sparsifying)
    end    
%     if (gMRA.Sizes(node)>gMRA.opts.ManifoldDimension) && (size(Y,1)>10),
%         gMRA.ScalFuns{node} = ksvd(struct('data', Y', 'Tdata', gMRA.opts.ManifoldDimension, 'initdict', gMRA.ScalFuns{node}, 'iternum', 10));   %KSVD
%         param.K=min([ceil(size(Y,1)/2),2*size(gMRA.ScalFuns{node},2)]);  % learns a dictionary with 100 elements
%         param.lambda= 5;
%         param.numThreads=-1; % number of threads
%         param.approx=4.0;   %MM: no idea what this is for        
%         param.iter = -5; % let's wait only 20 seconds
%         param.mode = 0;
%         param.D = gMRA.ScalFuns{node};
%         D = mexTrainDL(Y',param);
%         param.mode = 1;
%         param.lambda = norm(Y'-gMRA.ScalFuns{node}*gMRA.ScalFuns{node}'*Y');
%         newcoeffs = mexLasso(Y',D,param);
%         newcoeffs2 = mexLasso(Y',gMRA.ScalFuns{node},param);
%         gMRA.ScalFuns{node} = D;
%    end    
end

return;

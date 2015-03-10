function gW = construct_economicGeometricWavelets( gW )

% Minimal encoding-cost pruning
gW.opts.splitting = false;
gW.opts.avoidLeafnodePhi = false;
gW.opts.addTangentCorrections = false;

if gW.opts.ManifoldDimension > 0, % conflict
    gW.opts.ManifoldDimension = 0;
    warning('Manifold Dimension is NOT used and has been set to zero in order to allow for locally adaptive dimensions!'); %#ok<WNTAG>
end

% Encoding costs by PCA plane at each node to be within given precision
nAllNets = numel(gW.cp);
gW.epsEncodingCosts = zeros(1,nAllNets);   

%% construct the scaling functions only at the leaf nodes
gW = construct_vNets_leafNodes( gW );

%%
flags = ones(1, nAllNets); % indicator of whether to keep (1) or remove (-1) a node

j = max(gW.Scales)-1; % current scale
n = 1; % at least 1 leaf node at scale J. It counts the number of nodes whose parents will need to be processed. When n=0, we will stop.

while j>0 && n>0
    
    nodes = find(gW.Scales == j);
    n = sum(gW.isaleaf(nodes)); % initiaized to be the number of leaf nodes at this scale
    
    nodes = nodes(~gW.isaleaf(nodes)); % check non-leaf nodes below
    
    for i = 1:length(nodes)
        
        node = nodes(i);
        children = find(gW.cp == node);
        
        if numel(children)>1 % multiple children, construct local geometric wavelets and compare among the different encoding costs
            
            gW.PointsInNet{node} = [gW.PointsInNet{children}];
            gW = local_SVD_analysis(gW,node);
                   
            cost_p = gW.epsEncodingCosts(node); % parent only, computed in local_SVD_analysis above
            cost_c = sum(gW.epsEncodingCosts(children)); % children only, already optimized in previous loop                     
            [gW,cost_w,optScalFun] = optimalWavelets(gW, node, children); % wavelet method, computed in the subfunction below
            
            [minCost, bestMethod] = min([cost_p, cost_c, cost_w]);
            
            switch bestMethod
                
                case 1 % parent-only encoding is the best, then remove its offspring. This current node becomes a leaf node
                    
                    n = n+1; % need to check the parent of this node next time
                    
                    flags(get_offspring(gW.cp,node)) = -1;
                    gW.cp(children) = -1;
                    gW.IniLabels(gW.PointsInNet{node}) = node;
                    
                case 2 % children-only requires the least space, then separate children from the tree and delete the current node
                    
                    flags(node) = -1;                   
                    gW.cp(children) = 0;
                    gW.cp(node) = -1;
                                       
                case 3 % parent+wavelets is the best 
                    
                    n = n+1; % need to check the parent of node next time
                    gW.epsEncodingCosts(node) = minCost;
                    gW.ScalFuns{node} = optScalFun;
                      
                    % compute wavelet translations
                    for c = 1:length(children)
                        
                        gW.WavConsts{children(c)} = gW.Centers{children(c)} - gW.Centers{node};
                        
                        if ~gW.opts.sparsifying
                            gW.WavConsts{children(c)} = gW.WavConsts{children(c)} - (gW.WavConsts{children(c)}*gW.ScalFuns{node})*gW.ScalFuns{node}';
                        else
                            gW.WavConsts{children(c)} = gW.WavConsts{children(c)} - (gW.ScalFuns{node}\gW.WavConsts{children(c)}')' * gW.ScalFuns{node}';
                        end
                        
                    end
                    
            end
       
        elseif numel(children) == 1 % single child, remove this node and connect children directly to the grandparent
            
            n = n+1; % need to check the parent of node next round
            
            flags(node) = -1;
            gW.cp(children) = gW.cp(node);
            gW.cp(node) = -1;
                       
        else  %isempty(children) due to case 2 above (originally not a leaf node), will delete this node
            
            flags(node) = -1;
            gW.cp(node) = -1;
            
        end
        
    end
    
    j = j-1; 
    
end

% stopped before it reached the root of the tree
if j>0
    
    nodes = find(gW.Scales <= j);
    flags(nodes) = -1;
    gW.cp(nodes) = -1;
    
end

%%
gW = get_subtree(gW, flags);
%figure; treeplot(gW.cp); title 'metis tree (after pruning)';

%%
roots = find(gW.cp==0);
gW.WavBases(roots) = gW.ScalFuns(roots);
gW.WavConsts(roots) = gW.Centers(roots);
gW.WavSingVals(roots) = gW.Sigmas(roots);

return

function [gW,optCost, optScalFun] = optimalWavelets(gW,node, children)


allFineBases = [gW.ScalFuns{children}];

nChildren = numel(children);

%% first put empty scaling function at parent

parentDimension = 0;
allFineBasesOrthoProj=allFineBases;

[commonBasisPerp,S]     = svd(allFineBasesOrthoProj, 'econ');
intersectionDimension   = sum(diag(S)>sqrt(nChildren)-gW.opts.threshold2);
commonBasisPerp         = commonBasisPerp(:,1:intersectionDimension);

%gW.WavSingVals{node} = diag(S);

if gW.opts.addTangentialCorrections
    cost_w = sum(gW.epsEncodingCosts(children)) + intersectionDimension*(gW.Sizes(node)+gW.opts.AmbientDimension) + gW.opts.AmbientDimension;
else
    cost_w = sum(gW.epsEncodingCosts(children)) + intersectionDimension*(gW.Sizes(node)+gW.opts.AmbientDimension) + gW.opts.AmbientDimension*(nChildren+1);
end

for c = 1:nChildren
             
    remainingComponents = gW.ScalFuns{children(c)} - commonBasisPerp*(commonBasisPerp'*gW.ScalFuns{children(c)});
    [remainingBasisPerp,S] = svd(remainingComponents, 'econ');
    remDim = sum(diag(S)>gW.opts.threshold1);
    gW.WavBases{children(c)} = remainingBasisPerp(:,1:remDim);
    gW.WavSingVals{children(c)} = diag(S(1:remDim,1:remDim));

    if gW.opts.addTangentialCorrections
        cost_w = cost_w - size(gW.ScalFuns{children(c)},2)*gW.Sizes(children(c)) + size(gW.WavBases{children(c)},2)*(gW.Sizes(children(c))+gW.opts.AmbientDimension);
    else
        cost_w = cost_w - (size(gW.ScalFuns{children(c)},2)-size(gW.WavBases{children(c)},2))*(gW.Sizes(children(c))+gW.opts.AmbientDimension);
    end
    
end

optDim = parentDimension;
optCommonBasisPerp = commonBasisPerp;
optCost = cost_w;

%% increasing dimension of scaling funtion put at parent and choose the best

% col = 0;
% for c = 1:nChildren
%     allFineBasesOrthoProj(:, col+1:col+size(gW.ScalFuns{children(c)}, 2))  = allFineBases(:, col+1:col+size(gW.ScalFuns{children(c)}, 2))*diag(gW.Sigmas{children(c)}(1:size(gW.ScalFuns{children(c)}, 2)));
%     col = col+size(gW.ScalFuns{children(c)}, 2);
% end

for parentDimension = 1:size(gW.ScalFuns{node},2)-1
    
    allFineBasesOrthoProj = allFineBasesOrthoProj -  gW.ScalFuns{node}(:, parentDimension) * (gW.ScalFuns{node}(:, parentDimension)'*allFineBasesOrthoProj);
    
    tempWavBases = cell(1,nChildren);
    col = 0;
    for c = 1:nChildren
        
        [U,S] = svd(allFineBasesOrthoProj(:, col+1:col+size(gW.ScalFuns{children(c)}, 2)), 'econ');
        tempWavBases{c} = U(:,1:sum(diag(S)>gW.opts.threshold1));
        
        col = col+size(gW.ScalFuns{children(c)}, 2);
    
    end

    [commonBasisPerp,S] = svd([tempWavBases{:}], 'econ');
    intersectionDimension = sum(diag(S)>sqrt(nChildren)-gW.opts.threshold2);
    commonBasisPerp = commonBasisPerp(:,1:intersectionDimension);
    
    if gW.opts.addTangentialCorrections
        cost_w = sum(gW.epsEncodingCosts(children)) + (intersectionDimension+parentDimension)*(gW.Sizes(node)+gW.opts.AmbientDimension) + gW.opts.AmbientDimension;
    else
        cost_w = sum(gW.epsEncodingCosts(children)) + (intersectionDimension+parentDimension)*(gW.Sizes(node)+gW.opts.AmbientDimension) + gW.opts.AmbientDimension*(nChildren+1);
    end
    
    for c = 1:nChildren
               
        remainingComponents = tempWavBases{c} - commonBasisPerp*(commonBasisPerp'*tempWavBases{c});
        
        [remainingBasisPerp,S] = svd(remainingComponents, 'econ');
        tempWavBases{c} = remainingBasisPerp(:,1:sum(diag(S)>gW.opts.threshold1));
        
        if gW.opts.addTangentialCorrections
            cost_w = cost_w - size(gW.ScalFuns{children(c)},2)*gW.Sizes(children(c)) + size(tempWavBases{c},2)*(gW.Sizes(children(c))+gW.opts.AmbientDimension);
        else
            cost_w = cost_w - (size(gW.ScalFuns{children(c)},2)-size(tempWavBases{c},2))*(gW.Sizes(children(c))+gW.opts.AmbientDimension);
        end
        
    end
    
    if optCost > cost_w,
        
        optCost = cost_w;
        optDim = parentDimension;
        optCommonBasisPerp = commonBasisPerp;
        gW.WavBases(children) = tempWavBases;
        
    end
    
end

%%
optScalFun = [gW.ScalFuns{node}(:, 1:optDim) optCommonBasisPerp];

return;

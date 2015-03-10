function gMRA = construct_pruningGeometricWavelets( gMRA )

% Encoding costs by PCA plane at each node to be within given precision
nAllNets = numel(gMRA.cp);
gMRA.epsEncodingCosts = zeros(1,nAllNets);   
gMRA.dictEncodingCosts = zeros(1,nAllNets);  

%% construct the scaling functions only at the leaf nodes
gMRA = construct_vNets_leafNodes( gMRA );

%%
flags = ones(1, nAllNets); % indicator of whether to keep (1) or remove (-1) a node

j = max(gMRA.Scales)-1; % current scale
n = 1; % at least 1 leaf node at scale J. It counts the number of nodes whose parents will need to be processed. When n=0, we will stop.

while j>0 && n>0
    
    nodes = find(gMRA.Scales == j);
    n = sum(gMRA.isaleaf(nodes)); % initiaized to be the number of leaf nodes at this scale
    
    nodes = nodes(~gMRA.isaleaf(nodes)); % check non-leaf nodes below
    
    for i = 1:length(nodes)
        
        node = nodes(i);
        children = find(gMRA.cp == node);
        
        if numel(children)>1 % multiple children, construct local geometric wavelets and compare among the different encoding costs
            
            gMRA.PointsInNet{node} = [gMRA.PointsInNet{children}];
            gMRA = local_SVD_analysis(gMRA,node);
                   
            cost_p = gMRA.dictEncodingCosts(node); % parent only, computed in local_SVD_analysis above
            cost_c = sum(gMRA.dictEncodingCosts(children)); % children only, already optimized in previous loop                     
            
            [gMRA,cost_w,optScalFun] = optimalWavelets(gMRA, node, children); % wavelet method, computed in the subfunction below
            
            [minCost, bestMethod] = min([cost_p, cost_c, cost_w]);
            
            switch bestMethod
                
                case 1 % parent-only encoding is the best, then remove its offspring. This current node becomes a leaf node
                    
                    n = n+1; % need to check the parent of this node next time
                    
                    flags(get_offspring(gMRA.cp,node)) = -1;
                    gMRA.cp(children) = -1;
                    gMRA.IniLabels(gMRA.PointsInNet{node}) = node;
                    
                case 2 % children-only requires the least space, then separate children from the tree and delete the current node
                    
                    flags(node) = -1;                   
                    gMRA.cp(children) = 0;
                    gMRA.cp(node) = -1;
                                       
                case 3 % parent+wavelets is the best 
                    
                    n = n+1; % need to check the parent of node next time
                    gMRA.dictEncodingCosts(node) = minCost;
                    gMRA.ScalFuns{node} = optScalFun;
                      
                    % compute wavelet translations
                    for c = 1:length(children)
                        
                        gMRA.WavConsts{children(c)} = gMRA.Centers{children(c)} - gMRA.Centers{node};
                        
                        if ~gMRA.opts.sparsifying
                            gMRA.WavConsts{children(c)} = gMRA.WavConsts{children(c)} - gMRA.ScalFuns{node}*(gMRA.ScalFuns{node}'*gMRA.WavConsts{children(c)});
                        else
                            gMRA.WavConsts{children(c)} = gMRA.WavConsts{children(c)} - gMRA.ScalFuns{node}*(gMRA.ScalFuns{node}\gMRA.WavConsts{children(c)});
                        end
                        
                    end
                    
            end
       
        elseif numel(children) == 1 % single child, remove this node and connect children directly to the grandparent
            
            n = n+1; % need to check the parent of node next round
            
            flags(node) = -1;
            gMRA.cp(children) = gMRA.cp(node);
            gMRA.cp(node) = -1;
                       
        else  %isempty(children) due to case 2 above (originally not a leaf node), will delete this node
            
            flags(node) = -1;
            gMRA.cp(node) = -1;
            
        end
        
    end
    
    j = j-1; 
    
end

% stopped before it reached the root of the tree
if j>0
    
    nodes = find(gMRA.Scales <= j);
    flags(nodes) = -1;
    gMRA.cp(nodes) = -1;
    
end

%%
gMRA = get_subtree(gMRA, flags);
%figure; treeplot(gW.cp); title 'metis tree (after pruning)';

%%
roots = find(gMRA.cp==0);
gMRA.WavBases(roots) = gMRA.ScalFuns(roots);
gMRA.WavConsts(roots) = gMRA.Centers(roots);
gMRA.WavSingVals(roots) = gMRA.Sigmas(roots);

return

function [gMRA,optCost,optScalFun] = optimalWavelets(gMRA,node,children)

allFineBases = [gMRA.ScalFuns{children}];
nChildren = numel(children);

%% first put empty scaling function at parent

parentDimension = 0;
allFineBasesOrthoProj=allFineBases;

[commonBasisPerp,S]     = svd(allFineBasesOrthoProj, 'econ');
intersectionDimension   = sum(diag(S)>sqrt(nChildren)-gMRA.opts.threshold2);
commonBasisPerp         = commonBasisPerp(:,1:intersectionDimension);

%gW.WavSingVals{node} = diag(S);

if gMRA.opts.addTangentialCorrections
    %cost_w = sum(gW.epsEncodingCosts(children)) + intersectionDimension*(gW.Sizes(node)+gW.opts.AmbientDimension) + gW.opts.AmbientDimension;
    cost_w = sum(gMRA.dictEncodingCosts(children)) + intersectionDimension*(gMRA.Sizes(node)+gMRA.opts.AmbientDimension);
else
    %cost_w = sum(gW.epsEncodingCosts(children)) + intersectionDimension*(gW.Sizes(node)+gW.opts.AmbientDimension) + gW.opts.AmbientDimension*(nChildren+1);
    cost_w = sum(gMRA.dictEncodingCosts(children)) + intersectionDimension*(gMRA.Sizes(node)+gMRA.opts.AmbientDimension) + gMRA.opts.AmbientDimension*nChildren;
end

for c = 1:nChildren
             
    remainingComponents = gMRA.ScalFuns{children(c)} - commonBasisPerp*(commonBasisPerp'*gMRA.ScalFuns{children(c)});
    [remainingBasisPerp,S] = svd(remainingComponents, 'econ');
    remDim = sum(diag(S)>gMRA.opts.threshold1);
    gMRA.WavBases{children(c)} = remainingBasisPerp(:,1:remDim);
    gMRA.WavSingVals{children(c)} = diag(S(1:remDim,1:remDim));

    if gMRA.opts.addTangentialCorrections
        %cost_w = cost_w - size(gW.ScalFuns{children(c)},2)*gW.Sizes(children(c)) + size(gW.WavBases{children(c)},2)*(gW.Sizes(children(c))+gW.opts.AmbientDimension);
        cost_w = cost_w + size(gMRA.WavBases{children(c)},2)*gMRA.opts.AmbientDimension;
    else
        %cost_w = cost_w - (size(gW.ScalFuns{children(c)},2)-size(gW.WavBases{children(c)},2))*(gW.Sizes(children(c))+gW.opts.AmbientDimension);
        cost_w = cost_w + (size(gMRA.WavBases{children(c)},2)-size(gMRA.ScalFuns{children(c)},2))*(gMRA.opts.AmbientDimension);
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

for parentDimension = 1:size(gMRA.ScalFuns{node},2)-1
    
    allFineBasesOrthoProj = allFineBasesOrthoProj -  gMRA.ScalFuns{node}(:, parentDimension) * (gMRA.ScalFuns{node}(:, parentDimension)'*allFineBasesOrthoProj);
    
    tempWavBases = cell(1,nChildren);
    col = 0;
    for c = 1:nChildren
        
        [U,S] = svd(allFineBasesOrthoProj(:, col+1:col+size(gMRA.ScalFuns{children(c)}, 2)), 'econ');
        tempWavBases{c} = U(:,1:sum(diag(S)>gMRA.opts.threshold1));
        
        col = col+size(gMRA.ScalFuns{children(c)}, 2);
    
    end

    [commonBasisPerp,S] = svd([tempWavBases{:}], 'econ');
    intersectionDimension = sum(diag(S)>sqrt(nChildren)-gMRA.opts.threshold2);
    commonBasisPerp = commonBasisPerp(:,1:intersectionDimension);
    
    if gMRA.opts.addTangentialCorrections
        %cost_w = sum(gW.epsEncodingCosts(children)) + (intersectionDimension+parentDimension)*(gW.Sizes(node)+gW.opts.AmbientDimension) + gW.opts.AmbientDimension;
        cost_w = sum(gMRA.dictEncodingCosts(children)) + (intersectionDimension+parentDimension)*(gMRA.opts.AmbientDimension);
    else
        %cost_w = sum(gW.epsEncodingCosts(children)) + (intersectionDimension+parentDimension)*(gW.Sizes(node)+gW.opts.AmbientDimension) + gW.opts.AmbientDimension*(nChildren+1);
        cost_w = sum(gMRA.dictEncodingCosts(children)) + (intersectionDimension+parentDimension)*(gMRA.opts.AmbientDimension) + gMRA.opts.AmbientDimension*nChildren;
    end
    
    for c = 1:nChildren
               
        remainingComponents = tempWavBases{c} - commonBasisPerp*(commonBasisPerp'*tempWavBases{c});
        
        [remainingBasisPerp,S] = svd(remainingComponents, 'econ');
        tempWavBases{c} = remainingBasisPerp(:,1:sum(diag(S)>gMRA.opts.threshold1));
        
        if gMRA.opts.addTangentialCorrections
            %cost_w = cost_w - size(gW.ScalFuns{children(c)},2)*gW.Sizes(children(c)) + size(tempWavBases{c},2)*(gW.Sizes(children(c))+gW.opts.AmbientDimension);
            cost_w = cost_w + size(tempWavBases{c},2)*gMRA.opts.AmbientDimension;
        else
            %cost_w = cost_w - (size(gW.ScalFuns{children(c)},2)-size(tempWavBases{c},2))*(gW.Sizes(children(c))+gW.opts.AmbientDimension);
            cost_w = cost_w - (size(gMRA.ScalFuns{children(c)},2)-size(tempWavBases{c},2))*gMRA.opts.AmbientDimension;
        end
        
    end
    
    if optCost > cost_w,
        
        optCost = cost_w;
        optDim = parentDimension;
        optCommonBasisPerp = commonBasisPerp;
        gMRA.WavBases(children) = tempWavBases;
        
    end
    
end

%%
optScalFun = [gMRA.ScalFuns{node}(:, 1:optDim) optCommonBasisPerp];

return;

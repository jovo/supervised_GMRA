function gW = construct_orthogonalGeometricWavelets( gW )

% build the tree, and compute svd only at the roots
root = find(gW.cp==0);
gW  = construct_vNets_NoSVD(gW,root);       
gW  = local_SVD_analysis(gW,root);

% initialization
gW.WavBases(root) = gW.ScalFuns(root);
gW.WavConsts(root) = gW.Centers(root);
gW.WavSingVals(root) = gW.Sigmas(root);

% vector of indicators (whether to keep (1) or remove (-1) the node) 
flags = ones(1, numel(gW.cp)); 

%%
J = max(gW.Scales);
nAllNodes = length(gW.cp);

% scale 1
j = 1;
parentNodes = root; % set of nodes whose children will be processed (i.e., computing wavelet bases)

while j<J && ~isempty(parentNodes);
        
    childrenNodes = zeros(1,nAllNodes); % collecting nodes for next round
    
    for i = 1:length(parentNodes)
        
        node = parentNodes(i);        

        if ~gW.isaleaf(node)
            
            children = find(gW.cp==node);
            
            for c = 1:length(children)
                gW = local_SVD_analysis(gW,children(c));
            end
            
            gW = construct_localGeometricWavelets(gW,node,children);
            
            ancestors = get_ancestors(gW.cp, node);
           
            for c = 1:length(children)
                
                cumDict = [gW.WavBases{[children(c) node ancestors]}]; % orthogonal matrix by induction
                
                Y = bsxfun(@minus, gW.X(:,gW.PointsInNet{children(c)}), gW.Centers{children(c)});
                
                Y_proj = cumDict'*Y;
                approxErr = sum(Y.^2, 1) - sum((Y_proj).^2,1);
                
                if strcmpi(gW.opts.errorType, 'absolute')
                    approxErr = sqrt(mean(approxErr));
                else
                    sigs = svd(Y_proj);
                    approxErr = sum(approxErr)/(sum(sigs.^2)+sum(approxErr));
                end
                
                if approxErr <= gW.opts.precision % already within precision
                    offspring = get_offspring(gW.cp, children(c));
                    flags(offspring) = -1;
                    gW.IniLabels(gW.PointsInNet{children(c)}) = children(c);
                else
                    childrenNodes(children(c)) = 1;
                end
                
            end
            
        end
        
    end
    
    j = j+1;
    parentNodes = find(childrenNodes>0);
    
end

%%
gW = get_subtree(gW, flags);

return;
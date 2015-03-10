function construct_GMRA_recursively

% plain construction of geometric wavelets, implemented in a recursive way

global gW

roots = find(gW.cp==0);
for r = 1:length(roots)  
    construct_GMRA_oneTree(roots(r)); % one tree each time
end

%%
roots = find(gW.cp==0);
gW.WavBases(roots) = gW.ScalFuns(roots);
gW.WavConsts(roots) = gW.Centers(roots);
gW.WavSingVals(roots) = gW.Sigmas(roots);

return

function construct_GMRA_oneTree(node)

global gW

children = find(gW.cp == node);

if isempty(children) % leaf node
    
    gW.PointsInNet{node} = find(gW.IniLabels == node);  
        
else % not a leaf node, then go down to the children
    
    for c = 1:length(children)
        construct_GMRA_oneTree(children(c));
    end
    
    gW.PointsInNet{node} = [gW.PointsInNet{children}];
    
end

%% compute local center and basis

local_SVD_analysis(node);

%% construct wavlets between node and its children

if ~isempty(children)  
     construct_localGeometricWavelets(node, children)
end

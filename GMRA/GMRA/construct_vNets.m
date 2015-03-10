function construct_vNets(gW,node)

% This function performs local svd analysis at the input node by recursively
% working on all its children.

if gW.isaleaf(node) % leaf node
    
    gW.PointsInNet{node} = find(gW.IniLabels == node);
    
else % not leaf node, then go down to the children
    
    children = find(gW.cp == node);

    for c = 1:length(children),
        construct_vNets(children(c)); 
    end;
    
    gW.PointsInNet{node}  = [gW.PointsInNet{children}];
    
end

%% compute local center and basis

local_SVD_analysis(node);

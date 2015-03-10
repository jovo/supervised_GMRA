function gW = construct_vNets_NoSVD(gW,node)

% This function forms a subtree at the input node by recursively
% working on all its children.

if gW.isaleaf(node) % leaf node
    
    gW.PointsInNet{node} = find(gW.IniLabels == node);
    
else % not leaf node, then go down to the children

    children = find(gW.cp == node);

    for c = 1:length(children),
        gW = construct_vNets_NoSVD(gW,children(c)); 
    end;
    
    gW.PointsInNet{node}  = [gW.PointsInNet{children}];
    
end

return;

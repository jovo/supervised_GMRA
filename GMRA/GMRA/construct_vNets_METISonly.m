function gW = construct_vNet_METISonly(gW,node)

% This function forms a subtree at the input node by recursively
% working on all its children.

if gW.isaleaf(node) % leaf node
    
    gW.PointsInNet{node} = find(gW.IniLabels == node);
    PointsInThisNode    = gW.PointsInNet{node};
    gW.Sizes(node)    = numel(PointsInThisNode);
    lPtsInNode          = gW.X(:,PointsInThisNode);
    gW.Centers{node}  = mean(lPtsInNode,2);
    
else % not leaf node, then go down to the children
    
    children = find(gW.cp == node);

    for c = 1:length(children),
        gW = construct_vNets_METISonly(gW,children(c)); 
    end;
    
    gW.PointsInNet{node}  = [gW.PointsInNet{children}];
    PointsInThisNode    = gW.PointsInNet{node};
    gW.Sizes(node)    = numel(PointsInThisNode);
    lPtsInNode          = gW.X(:,PointsInThisNode);
    gW.Centers{node}  = mean(lPtsInNode,2);
end

return;

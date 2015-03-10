function gW = construct_vNets_leafNodes( gW )


nLeafnodes = length(gW.LeafNodes);

for i = 1:nLeafnodes,    
    node = gW.LeafNodes(i);                                     % Get the index of the current node
    gW.PointsInNet{node} = find(gW.IniLabels == node);          % Get the set of points in the neighborhood
    gW = local_SVD_analysis(gW,node);                           % Perform local SVD analysis
end

gW = trimming_small_leaves( gW );

return;

function scales = compute_scales(cp)

% computes the scales of the nodes in the metis tree. 
% The root node has scale 1 and 
% the leaf nodes have the largest (positive) scales.

nAllNets = numel(cp);
currentNodes = 1:nAllNets;

scales = zeros(1, nAllNets);
while any(currentNodes)
    
    nonzeroNodes = (currentNodes > 0);
    scales(nonzeroNodes) = scales(nonzeroNodes) + 1;
    currentNodes(nonzeroNodes) = cp(currentNodes(nonzeroNodes));

end
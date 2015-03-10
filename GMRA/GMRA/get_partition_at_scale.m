function idxs = get_partition_at_scale( GMRA, j )

%
% Returns a cover of the set of leaves made by nodes at scale j, 
% plus any additional leaf that may be needed at scale coarser than j
% 

idxs = sort([find(GMRA.Scales == j) GMRA.LeafNodes(GMRA.Scales(GMRA.LeafNodes)<j)], 'ascend');

return;
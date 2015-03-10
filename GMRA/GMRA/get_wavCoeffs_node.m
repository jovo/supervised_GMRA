function W = get_wavCoeffs_node(GMRA, Data, n)

% get the wavelet coefficients of all points in the node n

[~, offspringLeafNodes] = get_offspring(GMRA.cp, n);

imap(GMRA.LeafNodes) = 1:length(GMRA.LeafNodes);
W = cat(1, Data.CelWavCoeffs{imap(offspringLeafNodes), GMRA.Scales(n)});

return;
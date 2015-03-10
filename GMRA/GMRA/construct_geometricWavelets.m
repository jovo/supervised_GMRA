function gW = construct_geometricWavelets( gW )

%% Construction of a GMRA

% Construct the leaf nodes
gW = construct_vNets_leafNodes( gW );

J = max(gW.Scales);

% Go bottom up through coarser scales
for j = J-1:-1:1
    % Find nodes at scale j
    nodes = find((gW.Scales==j) & (~gW.isaleaf));
    % Go through the nodes at scale j
    for k = 1:length(nodes)
        node = nodes(k);
        children = find(gW.cp==node);
        gW.PointsInNet{node} = [gW.PointsInNet{children}];
        % Construct the scaling functions at node (j,k)
        gW = local_SVD_analysis(gW,node);
        % Construct the wavelets (j+1,children(j,k))
        gW = construct_localGeometricWavelets(gW,node,children);
    end
end

for j = J:-1:1,
    gW.Radii_AvePerScale(j) = mean(gW.Radii(gW.Scales==j));
end;

% Wavelet bases at the root are just the scaling functions
roots = find(gW.cp==0);
gW.WavBases(roots)     = gW.ScalFuns(roots);
gW.WavConsts(roots)    = gW.Centers(roots);
gW.WavSingVals(roots)  = gW.Sigmas(roots);

return;

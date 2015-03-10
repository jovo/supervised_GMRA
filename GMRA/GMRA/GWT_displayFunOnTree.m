function GWT_displayFunOnTree(GWT, Data, f)

% f is a function of the nodes in the GWT tree, and possibly of Data

N           = numel(GWT.IniLabels);
nAllNets    = numel(GWT.Scales);
J           = max(GWT.Scales);
matF        = zeros(J, N);

for n = 1:nAllNets
    matF(GWT.Scales(n), GWT.PointsInNet{n}) = f(GWT,n,Data);
end

matF = matF(:, [GWT.PointsInNet{GWT.LeafNodes}]);

figure;DrawGWTWavSubFcn( GWT, matF, 'f');

return;

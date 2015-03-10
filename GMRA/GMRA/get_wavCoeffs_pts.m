function W = get_wavCoeffs_pts(gW,Data,pts,j)

% get the wavelet coefficients of pts at scale j. 
% If j is not provided, then all scales are returned.

ptsOrderInLeafNodes = [gW.PointsInNet{gW.LeafNodes}];
N = length(ptsOrderInLeafNodes);
imap = zeros(1, N);
imap(ptsOrderInLeafNodes) = 1:N;

if nargin<4,
    W = Data.MatWavCoeffs(imap(pts), :);
else
    W = Data.MatWavCoeffs(imap(pts), sum(Data.maxWavDims(1:j-1))+1 : sum(Data.maxWavDims(1:j)));
end
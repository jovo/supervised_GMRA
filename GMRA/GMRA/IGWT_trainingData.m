function [Projections, tangentialCorrections] = IGWT_trainingData(gW, CelWavCoeffs)

% Inverse Geometric Wavelet Transform
%
% Input: 
%   gW: structure of wavelet bases and translations
%       .WavBases: wavelet basis
%       .WavConsts: associated translations
%   CelWavCoeffs: cell array of wavelet coefficients
%
% Output:
%  Projections: D-N-J matirx of recovered data at all scales
%  TangentialCorrections: D-N-J array of tangential corrections

N = numel(gW.IniLabels);
J = max(gW.Scales); % number of scales

nLeafNodes = numel(gW.LeafNodes);

x_mat = zeros(gW.opts.AmbientDimension, N, J); % wavelets at all J scales

if nargout>1,
    tangentialCorrections = zeros(gW.opts.AmbientDimension, N, J);
end

%%
for i = 1:nLeafNodes
    
    net    = gW.LeafNodes(i);
    netPts = find(gW.IniLabels == net);
    nPts   = length(netPts);
    
    j_max = gW.Scales(net); % number of scales involved
    chain = dpath(gW.cp, net);
   
    for j = j_max:-1:1
        
        x_mat(:,netPts,j) = repmat(gW.WavConsts{chain(j)}, 1, nPts);
        
        if ~isempty(gW.WavBases{chain(j)})
            x_mat(:,netPts,j) = x_mat(:,netPts,j) + gW.WavBases{chain(j)}*CelWavCoeffs{i,j}';
        end
        
    end
        
    if gW.opts.addTangentialCorrections     % x_mat is corrected by adding tangential corrections
        
        for j = j_max-1:-1:2
            
            x_TC = gW.ScalFuns{chain(j-1)}*(gW.ScalFuns{chain(j-1)}'*sum(x_mat(:,netPts,j+1:j_max),3));
            x_mat(:,netPts,j) = x_mat(:,netPts,j) - x_TC;
            
            if nargout>1,
                tangentialCorrections(:,netPts, j) = x_TC;
            end
            
        end
        
    end
    
end

% 
Projections = cumsum(x_mat,3);

return;
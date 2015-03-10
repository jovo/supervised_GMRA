function [Projections, tangentialCorrections] = IGWT(gMRA, DataGWT)

% Inverse Geometric Wavelet Transform
%
% Input:
%   gMRA: structure of wavelet bases and translations
%   Data: output from FGWT_combined
%
% Output:
%  Projections: D-N-J matirx of recovered data at all scales
%  TangentialCorrections: D-N-J array of tangential corrections

%% Initialization
tangentialCorrections = [];
N = numel(DataGWT.leafNodeLabels);
J = max(gMRA.Scales); % number of scales
nLeafNodes = numel(gMRA.LeafNodes);

x_mat = zeros(gMRA.opts.AmbientDimension, N, J); % wavelets at all J scales

if gMRA.opts.addTangentialCorrections && nargout>1
    tangentialCorrections = zeros(gMRA.opts.AmbientDimension, N, J);
end

%%
for i = 1:nLeafNodes
    
    net    = gMRA.LeafNodes(i);
    netPts = find(DataGWT.leafNodeLabels == net);
    nPts   = length(netPts);
    
    if nPts>0,
        
        j_max = gMRA.Scales(net); % number of scales involved
        chain = dpath(gMRA.cp, net);
        
        for j = j_max:-1:1
            
            x_mat(:,netPts,j) = repmat(gMRA.WavConsts{chain(j)}, 1, nPts);
            
            if ~isempty(gMRA.WavBases{chain(j)})
                x_mat(:,netPts,j) = x_mat(:,netPts,j) + gMRA.WavBases{chain(j)}*DataGWT.CelWavCoeffs{i,j}';
            end
            
        end
        
        if gMRA.opts.addTangentialCorrections     % x_mat is corrected by adding tangential corrections
            
            for j = j_max-1:-1:2
                
                x_TC = gMRA.ScalFuns{chain(j-1)}*(gMRA.ScalFuns{chain(j-1)}'*sum(x_mat(:,netPts,j+1:j_max),3));
                x_mat(:,netPts,j) = x_mat(:,netPts,j) - x_TC;
                
                if nargout>1,
                    tangentialCorrections(:,netPts, j) = x_TC;
                end
                
            end
            
        end
        
    end
    
end

Projections = cumsum(x_mat,3);

return;
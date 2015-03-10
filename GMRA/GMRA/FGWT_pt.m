function Data = FGWT(gMRA, x)

%
% Fast Geometric Wavelets Transform
%
% Input:
%     gMRA: the geometric wavelets structure computed by the function
%         geometric_wavelets_transformatin. In fact, only the following fields
%         are needed:
%       .cp: the vector encoding the metis tree structure
%       .Scales: vector of scales of the nets
%       .Centers: the local centers of the nets
%       .WavBases: wavelet bases
%       .WavConsts: translations associated with the wavelet bases
%   x: a new point, D-by-1 vector
% 
% Output: 
%   Data: structure of the following fields:
%       .WavCoeffs: cell array of wavelet coefficients
%       .chain: path from the root to the leaf node along the tree 
%               This auxillary field can be used to extract from gW 
%               the sequences of wavelet bases and translations by
%               using gW.WavBases{chain} and gW.WavConsts{chain}

J = max(gMRA.Scales); % number of scales

Data            = struct();
Data.ScalCoeffs = cell(1,J); 
Data.WavCoeffs  = cell(1,J); 

%% Find the leaf node that is closest to x
net = find_nearest_leaf_node(gMRA, x); 

%% Compute transform bottom up
j = gMRA.Scales(net);

Data.chain = zeros(1,j);

if  j==1 % only one scale
    
    Data.chain(1)       = net;
    Data.WavCoeffs{1}   = (x-gMRA.Centers{net})' * gMRA.WavBases{net};
    Data.ScalCoeffs{1}  = Data.WavCoeffs{1};
    
else    % go bnottom up from the leaf to the root of the GMRA tree
    
    iFineNet    = net; % current scale
    iCoarseNet  = gMRA.cp(net); % next scale
    
    if ~gMRA.opts.orthogonalizing
        if gMRA.opts.avoidLeafnodePhi
            finestBasis = [gMRA.ScalFuns{iCoarseNet} gMRA.WavBases{iFineNet}];
        else
            finestBasis = gMRA.ScalFuns{iFineNet};
        end
        % x_J: finest approximation at scale J
        % x_j: scale j approximation for any j
%         Data.ScalCoeffs{j} = finestBasis'*(x-gMRA.Centers{iFineNet});
%         x_J = finestBasis*Data.ScalCoeffs{j} + gMRA.Centers{iFineNet}; % best approximation
        Data.ScalCoeffs{j} = (x-gMRA.Centers{iFineNet})'*finestBasis;
        x_J = finestBasis*Data.ScalCoeffs{j}' + gMRA.Centers{iFineNet}; % best approximation
        x_j = x_J;
    else
        %Data.ScalCoeffs{j} = (x-gMRA.Centers{iFineNet})*gMRA.ScalFuns{iFineNet};
        Data.ScalCoeffs{j} = (x-gMRA.Centers{iFineNet})'*gMRA.ScalFuns{iFineNet};
    end
    
    while j>1

        Data.chain(j) = iFineNet;
        
        if ~gMRA.opts.orthogonalizing
            wavelet = gMRA.WavConsts{iFineNet};
            
            if ~isempty(gMRA.WavBases{iFineNet})
                %Data.WavCoeffs{j} = gMRA.WavBases{iFineNet}'*(x_j-gMRA.Centers{iFineNet});
                Data.WavCoeffs{j} = (x_j-gMRA.Centers{iFineNet})'*gMRA.WavBases{iFineNet};
                wavelet = wavelet + gMRA.WavBases{iFineNet}*Data.WavCoeffs{j}';% gMRA.WavBases{iFineNet}*Data.WavCoeffs{j};
            end
            
            if gMRA.opts.addTangentialCorrections
                wavelet = wavelet - gMRA.ScalFuns{iCoarseNet}*gMRA.ScalFuns{iCoarseNet}'*(x_J-x_j);
            end
            
            x_j = x_j - wavelet;
            Data.ScalCoeffs{j-1} = (x_j-gMRA.Centers{iCoarseNet})' * gMRA.ScalFuns{iCoarseNet};
        else
            Data.ScalCoeffs{j-1} = (x-gMRA.Centers{iCoarseNet})' * gMRA.ScalFuns{iCoarseNet};
            Data.WavCoeffs{j} = (x-gMRA.Centers{iFineNet}) * gMRA.WavBases{iFineNet};   
        end
        
        j = j-1;
        iFineNet = iCoarseNet;
        iCoarseNet = gMRA.cp(iFineNet);
        
    end
    
    % j = 1
    Data.chain(1) = iFineNet;
    Data.WavCoeffs{1} = (x_j-gMRA.Centers{iFineNet})'*gMRA.WavBases{iFineNet};
    
end

return;

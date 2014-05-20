function DataGWT = FGWT(gMRA, X)

% FGWT - Forward Geometric Wavelet Transform on any input data
%
% INPUT
%   gMRA: the geometric multi-resolution analysis structure, as created by GMRA
%      X: D-by-N matrix of data points
%
% OUTPUT
%   DataGWT: a structure of the following fields:
%      .leafNodeLabels: an N-vector of indices of leaf nodes to which the
%                       input data are assigned by proximity
%      .LeafNodeSizes: number of points assigned to each leaf node
%      .dists2NearestLeafNodes: corresponding distances from data points to
%                               the centers of the nearest leaf nodes
%      .CelWavCoeffs: nLeafNodes by J matrix of cells, each cell (i,j)
%                     is a matrix containing in rows the wavelet coefficients of
%                     the points in the leaf node i and at the corresponding scale j
%      .CelScalCoeffs: similar to above, but stores scaling coefficients
%      .CelTangCoeffs: similar to above, but stores tangential coefficients
%      .MatWavCoeffs: matrix of wavelet coefficients, rows correspond to
%                     scales, columns correspond to points
%      .maxWavDims: vector of maximal wavelet dimension at each scale
%      .MatWavDims: N-by-J matrix of wavelet dimensions
%      .CoeffsCosts: cost in storing all nonzero wavelet coefficients

J = max(gMRA.Scales); % number of scales
nLeafNodes = numel(gMRA.LeafNodes);

%% Initialization 
DataGWT = struct();

% Find the nearest leaf to each data point
[~, nearestCenters, DataGWT.dists2NearestLeafNodes] = nrsearch(cat(2,gMRA.Centers{gMRA.LeafNodes}), X, 1, 0, [], struct('ReturnAsArrays',1));
DataGWT.leafNodeLabels = gMRA.LeafNodes(nearestCenters);
DataGWT.leafNodeSizes = zeros(1,nLeafNodes);

DataGWT.CelWavCoeffs = cell(nLeafNodes,J);
DataGWT.CelScalCoeffs = cell(nLeafNodes,J);
DataGWT.PointsInNet = cell(length(gMRA.cp),1);
if gMRA.opts.addTangentialCorrections,
    DataGWT.CelTangCoeffs = cell(nLeafNodes,J);
end

DataGWT.Cel_cpidx   = zeros(nLeafNodes,J);

DataGWT.PtIdxs = [];

%% Process each leaf node separately
for i = 1:nLeafNodes
    
    iFineNet = gMRA.LeafNodes(i);   
    netPts = find(DataGWT.leafNodeLabels == iFineNet);
    DataGWT.leafNodeSizes(i) = numel(netPts);
    DataGWT.PointsInNet{iFineNet} = netPts;
    DataGWT.PtIdxs = [DataGWT.PtIdxs,netPts];
    
    % not empty 
    if DataGWT.leafNodeSizes(i)>0
        
        j = gMRA.Scales(iFineNet); % current scale
        
        if j==1 % single-node tree
            
            DataGWT.CelWavCoeffs{i,1} = bsxfun(@minus, X(:,netPts), gMRA.Centers{iFineNet})'*gMRA.WavBases{iFineNet};
            DataGWT.Cel_cpidx(i,1) = iFineNet;
            DataGWT.CelScalCoeffs{i,1} = DataGWT.CelWavCoeffs{i,1};
       
        else
            
            % for scale at current leaf node
            iCoarseNet = gMRA.cp(iFineNet);
            DataGWT.PointsInNet{iCoarseNet} = [DataGWT.PointsInNet{iCoarseNet} netPts];
            
            if ~gMRA.opts.orthogonalizing && gMRA.opts.addTangentialCorrections
                if gMRA.opts.avoidLeafnodePhi
                    finestBasis = [gMRA.ScalFuns{iCoarseNet} gMRA.WavBases{iFineNet}];
                else
                    finestBasis = gMRA.ScalFuns{iFineNet};
                end
                ScalCoeffs = bsxfun(@minus, X(:,netPts), gMRA.Centers{iFineNet})'*finestBasis;
                Projections_jmax = bsxfun(@plus, finestBasis*ScalCoeffs', gMRA.Centers{iFineNet});
                DataGWT.CelScalCoeffs{i,j} = ScalCoeffs;
            else
                Projections_jmax = X(:,netPts);
                DataGWT.CelScalCoeffs{i,j} =  bsxfun(@minus, X(:,netPts), gMRA.Centers{iFineNet})'*gMRA.ScalFuns{iFineNet};
            end
            
            Projections = Projections_jmax;
            Wavelets = gMRA.WavConsts{iFineNet}(:,ones(DataGWT.leafNodeSizes(i),1));
            
            if ~isempty(gMRA.WavBases{iFineNet})
                DataGWT.CelWavCoeffs{i,j} = ComputeWaveletCoeffcients(bsxfun(@minus, Projections, gMRA.Centers{iFineNet}), ...
                    gMRA.WavBases{iFineNet}, gMRA.opts.sparsifying, gMRA.opts.precision*min(sqrt(sum(Projections.^2,1))));
                Wavelets = Wavelets + gMRA.WavBases{iFineNet}*DataGWT.CelWavCoeffs{i,j}';
            end
            
            if gMRA.opts.addTangentialCorrections
                DataGWT.CelTangCoeffs{i,j} = zeros(DataGWT.leafNodeSizes(i),size(gMRA.ScalFuns{iCoarseNet},2));
            end
            
            DataGWT.Cel_cpidx(i,j) = iFineNet;
            
            j = j-1;
            if ~gMRA.opts.orthogonalizing
                Projections = Projections - Wavelets;
            end
            
            % for scales between the leaf node and the root
            while j>1
                
                iFineNet = iCoarseNet;
                iCoarseNet = gMRA.cp(iFineNet);
                DataGWT.PointsInNet{iCoarseNet} = [DataGWT.PointsInNet{iCoarseNet} netPts];
            
                DataGWT.CelScalCoeffs{i,j} =  bsxfun(@minus, Projections, gMRA.Centers{iFineNet})'*gMRA.ScalFuns{iFineNet};
                Wavelets = gMRA.WavConsts{iFineNet}(:,ones(DataGWT.leafNodeSizes(i),1));
                
                if ~isempty(gMRA.WavBases{iFineNet})
                    DataGWT.CelWavCoeffs{i,j} = ComputeWaveletCoeffcients(bsxfun(@minus, Projections, gMRA.Centers{iFineNet}), ...
                        gMRA.WavBases{iFineNet}, gMRA.opts.sparsifying, gMRA.opts.threshold0(iFineNet)*min(sqrt(sum(Projections.^2,1))) );
                    Wavelets = Wavelets + gMRA.WavBases{iFineNet}*DataGWT.CelWavCoeffs{i,j}';
                end
                
                if gMRA.opts.addTangentialCorrections,
                    tangCoeffs = gMRA.ScalFuns{iCoarseNet}'*(Projections-Projections_jmax);
                    lTangentialCorrections = gMRA.ScalFuns{iCoarseNet}*tangCoeffs;
                    Wavelets = Wavelets + lTangentialCorrections;
                    DataGWT.CelTangCoeffs{i,j} = tangCoeffs';
                end
                
                if ~gMRA.opts.orthogonalizing
                    Projections = Projections - Wavelets;
                end
                
                DataGWT.Cel_cpidx(i,j) = iFineNet;
                
                j = j-1;
                
            end
            
            % at the root
            DataGWT.CelWavCoeffs{i,1}   = bsxfun(@minus, Projections, gMRA.Centers{iCoarseNet})'*gMRA.WavBases{iCoarseNet};
            DataGWT.Cel_cpidx(i,1)      = iCoarseNet;
            DataGWT.CelScalCoeffs{i,1}  = DataGWT.CelWavCoeffs{i,1};
            DataGWT.CelTangCoeffs{i,j}  = (Projections-Projections_jmax)'*gMRA.ScalFuns{iCoarseNet};
            
        end             
        
    else      
        
        j = gMRA.Scales(iFineNet); % current scale
        
        if j==1 % single-node tree
            
            DataGWT.Cel_cpidx(i,1) = iFineNet;
       
        else
            
            % for scale at current leaf node
            iCoarseNet = gMRA.cp(iFineNet);           
            
            DataGWT.Cel_cpidx(i,j) = iFineNet;
            
            j = j-1;
            
            % for scales between the leaf node and the root
            while j>1                
                iFineNet = iCoarseNet;
                iCoarseNet = gMRA.cp(iFineNet);
                DataGWT.Cel_cpidx(i,j) = iFineNet;                
                j = j-1;                
            end
            
            DataGWT.Cel_cpidx(i,1) = iCoarseNet;            
            
        end        
        
    end
    
end

%%
[DataGWT.MatWavCoeffs, DataGWT.maxWavDims, DataGWT.MatWavDims] = cell2mat_coeffs(DataGWT.CelWavCoeffs, cumsum(DataGWT.leafNodeSizes));
%[Data.MatScalCoeffs, Data.maxScalDims, Data.MatScalDims] = cell2mat_coeffs(Data.CelTangCoeffs);
DataGWT.CoeffsCosts = sum(sum(abs(DataGWT.MatWavCoeffs)>0));

return;


function wavCoeffs = ComputeWaveletCoeffcients(data, wavBases, sparsifying, epsilon)

if ~sparsifying,
    wavCoeffs = data'*wavBases;
else
    %wavCoeffs = transpose(mexLasso(data,wavBases,struct('iter', -5, 'mode', 1, 'lambda', 1e-3)));
    %wavCoeffs = full(omp(wavBases'*data,wavBases'*wavBases,5))';
    wavCoeffs = full(omp2(wavBases'*data,sum(data.*data),wavBases'*wavBases,epsilon))';
end

return;
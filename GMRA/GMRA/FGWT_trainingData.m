function Data = FGWT_trainingData(gW, X)

% Geometric Wavelet Transform using training data
%
% Input: 
%   gW: the geometric wavelets structure 
%    X: N-by-D matrix of data points
%
% OutPut:
%   Data: a structure of the following fields:
%      .CelWavCoeffs: nLeafNodes by J matrix of cells, each cell (i,j) 
%                     is a matrix containing in rows the wavelet coefficients of 
%                     the points in the leaf node i and at the corresponding scale j 

J = max(gW.Scales) % number of scales

nLeafNodes = numel(gW.LeafNodes);

%%
Data.CelWavCoeffs = cell(nLeafNodes,J);
Data.CelScalCoeffs = cell(nLeafNodes,J);
Data.Cel_cpidx = zeros(nLeafNodes,J);

if gW.opts.addTangentialCorrections,
    Data.CelTangCoeffs = cell(nLeafNodes,J);
end

for i = 1:nLeafNodes
    
    iFineNet = gW.LeafNodes(i);
    
    j = gW.Scales(iFineNet);

    netPts = find(gW.IniLabels == iFineNet);
    nPts = length(netPts);
    
    if j==1
        
        Data.CelWavCoeffs{i,1} = bsxfun(@minus, X(:,netPts), gW.Centers{iFineNet})'*gW.WavBases{iFineNet};
        Data.Cel_cpidx(i,1) = iFineNet;
        
    else % j>1
        %% at scale J 
        iCoarseNet = gW.cp(iFineNet); 
             
        if ~gW.opts.orthogonalizing && gW.opts.addTangentialCorrections
            if gW.opts.avoidLeafnodePhi
                finestBasis = [gW.ScalFuns{iCoarseNet} gW.WavBases{iFineNet}];
            else
                finestBasis = gW.ScalFuns{iFineNet};
            end
        
            ScalCoeffs = bsxfun(@minus, X(:,netPts), gW.Centers{iFineNet})'*finestBasis;
            Projections_J = bsxfun(@plus, finestBasis*ScalCoeffs', gW.Centers{iFineNet});
            Data.CelScalCoeffs{i,j} = ScalCoeffs;                       
        else           
            Projections_J = X(:,netPts);
            Data.CelScalCoeffs{i,j} =  bsxfun(@minus, X(:,netPts), gW.Centers{iFineNet})'*gW.ScalFuns{iFineNet};
        end
        
        Projections = Projections_J;    
        Wavelets = gW.WavConsts{iFineNet}(:,ones(nPts,1));
        
        if ~isempty(gW.WavBases{iFineNet})            
            Data.CelWavCoeffs{i,j} = computeWaveletCoeffcients(bsxfun(@minus, Projections, gW.Centers{iFineNet}), gW.WavBases{iFineNet}, gW.opts.sparsifying, gW.opts.precision*min(sqrt(sum(Projections.^2,1))));
            Wavelets = Wavelets + gW.WavBases{iFineNet}*Data.CelWavCoeffs{i,j}';
        end
        
        if gW.opts.addTangentialCorrections        
            Data.CelTangCoeffs{i,j} = zeros(nPts, size(gW.ScalFuns{iCoarseNet},2));
        end
        
        Data.Cel_cpidx(i,j) = iFineNet;
        
        j = j-1;

        if ~gW.opts.orthogonalizing
            Projections = Projections - Wavelets;
        end         
        
         %% for scales less than J
        while j>1
            
            iFineNet = iCoarseNet;
            iCoarseNet = gW.cp(iFineNet);
            
            Data.CelScalCoeffs{i,j} =  bsxfun(@minus, Projections, gW.Centers{iFineNet})'*gW.ScalFuns{iFineNet};
            
            Wavelets = gW.WavConsts{iFineNet}(:,ones(nPts,1));
            
            if ~isempty(gW.WavBases{iFineNet})
                Data.CelWavCoeffs{i,j} = computeWaveletCoeffcients(bsxfun(@minus, Projections, gW.Centers{iFineNet}), gW.WavBases{iFineNet}, gW.opts.sparsifying, gW.opts.threshold0(iFineNet)*min(sqrt(sum(Projections.^2,1))) );
                Wavelets = Wavelets + gW.WavBases{iFineNet}*Data.CelWavCoeffs{i,j}';
            end

            if gW.opts.addTangentialCorrections,                
                tangCoeffs = gW.ScalFuns{iCoarseNet}'*(Projections-Projections_J);
                lTangentialCorrections = gW.ScalFuns{iCoarseNet}*tangCoeffs;
                Wavelets = Wavelets + lTangentialCorrections;
                
                Data.CelTangCoeffs{i,j} = tangCoeffs;                
            end
            
            if ~gW.opts.orthogonalizing
                Projections = Projections - Wavelets;
            end
            
            Data.Cel_cpidx(i,j) = iFineNet;
             
            j = j-1;
            
        end
        
        %% at j = 1
        Data.CelWavCoeffs{i,1} = bsxfun(@minus, Projections, gW.Centers{iCoarseNet})'*gW.WavBases{iCoarseNet};
        
        Data.Cel_cpidx(i,1) = iCoarseNet;
        
    end
    
    Data.CelScalCoeffs{i,1} = Data.CelWavCoeffs{i,1};
            
end

%%
[Data.MatWavCoeffs, Data.maxWavDims, Data.MatWavDims] = cell2mat_coeffs(Data.CelWavCoeffs, cumsum(gW.Sizes(gW.LeafNodes)));
%[Data.MatScalCoeffs, Data.maxScalDims, Data.MatScalDims] = cell2mat_coeffs(Data.CelTangCoeffs);
Data.CoeffsCosts = sum(sum(abs(Data.MatWavCoeffs)>0));

return;


function wavCoeffs = computeWaveletCoeffcients(data, wavBases, sparsifying,epsilon)

if ~sparsifying,
    wavCoeffs = data'*wavBases;
else
    %wavCoeffs = transpose(mexLasso(data,wavBases,struct('iter', -5, 'mode', 1, 'lambda', 1e-3)));    
    %wavCoeffs = full(omp(wavBases'*data,wavBases'*wavBases,5))';    
    wavCoeffs = full(omp2(wavBases'*data,sum(data.*data),wavBases'*wavBases,epsilon))';    
end

return;


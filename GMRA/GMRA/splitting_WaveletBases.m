function gW = splitting_WaveletBases(gW, node, children)

if nargin<3, 
    children = find(gW.cp==node); 
end

nChildren = numel(children); 
if nChildren<=1; return; end
    
% compute the intersection of the wavelet subspaces W_{j,k}^\cap
allFineBasesPerpRenorm = [gW.WavBases{children}];
[commonBasisPerp,S] = svd(allFineBasesPerpRenorm, 'econ');
comDim = sum(diag(S)>sqrt(nChildren)-gW.opts.threshold2); % intersection dimension

if comDim>0 
    if ~gW.opts.mergePsiCapIntoPhi
        gW.WavDimsPsiCap(node) = comDim;
    else
        commonBasisPerp = commonBasisPerp(:,1:comDim);
        gW.ScalFuns{node} = [gW.ScalFuns{node}, commonBasisPerp];
        for c = 1:nChildren
            remainingComponents = gW.WavBases{children(c)} - commonBasisPerp*(commonBasisPerp'*gW.WavBases{children(c)});
            [remainingBasisPerp,S] = svd(remainingComponents, 'econ');
            
            remDim = sum(diag(S)>gW.opts.threshold1);
            %remDim = size(gW.WavBases{children(c)},2)-comDim;
            remainingBasisPerp = remainingBasisPerp(:, 1:remDim);
            
            gW.WavBases{children(c)} = remainingBasisPerp;
            gW.WavConsts{children(c)} = gW.WavConsts{children(c)} - ((gW.Centers{children(c)}-gW.Centers{node})*commonBasisPerp) * commonBasisPerp';
        end       
    end
end

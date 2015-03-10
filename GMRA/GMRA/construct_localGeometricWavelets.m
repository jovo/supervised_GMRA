function gW = construct_localGeometricWavelets(gW,node,children)

if nargin<3; children = find(gW.cp==node); end;

nChildren = numel(children);

%%
allFineBases = [gW.ScalFuns{children}];
% allFineBases = [];
% for c = 1:nChildren
%     allFineBases = [allFineBases gW.ScalFuns{children(c)}*diag(gW.Sigmas{children(c)}(1:size(gW.ScalFuns{children(c)},2)))];
% end

if ~gW.opts.orthogonalizing
    parentScalFuns = gW.ScalFuns{node};  
else   
    parentScalFuns = [gW.WavBases{[node get_ancestors(gW.cp, node)]}]; % orthogonal matrix by induction
end
    
if ~gW.opts.sparsifying
    allFineBasesPerp = allFineBases - parentScalFuns * (parentScalFuns'*allFineBases);
else
    allFineBasesPerp = allFineBases - parentScalFuns * (parentScalFuns\allFineBases);
end

%% Compute the wavelet subspaces W_{j+1,k'}
wavDims = zeros(1,nChildren);
col = 0;

for c = 1:nChildren
    
    [U,S] = svd(allFineBasesPerp(:, col+1:col+size(gW.ScalFuns{children(c)}, 2)), 'econ');
    wavDims(c) = sum(diag(S)>gW.opts.threshold1); % wavelet dimension
    if wavDims(c)>0
        gW.WavBases{children(c)} = U(:, 1:wavDims(c));   
        gW.WavSingVals{children(c)} = diag(S(1:wavDims(c), 1:wavDims(c)));
    end
    
    gW.WavConsts{children(c)}       = gW.Centers{children(c)} - gW.Centers{node};
    if gW.opts.sparsifying
        gW.WavConsts{children(c)}   = gW.WavConsts{children(c)} - parentScalFuns*(parentScalFuns\gW.WavConsts{children(c)});
    else
        gW.WavConsts{children(c)}   = gW.WavConsts{children(c)} - parentScalFuns*(parentScalFuns'*gW.WavConsts{children(c)});
    end
        
    col = col+size(gW.ScalFuns{children(c)}, 2);
    
end

%% Splitting of the wavelet subspaces for the children nodes, if requested
if gW.opts.splitting
    gW = splitting_WaveletBases(gW, node, children);
end

%% Sparsifying basis in the wavelet subspaces, if requested
if gW.opts.sparsifying
    gW = sparsifying_WaveletBases(gW, node, children);    
end

return;

function cost = computeDictionaryCost(gW,idxs)

if nargin<2, idxs = 1:length(gW.cp); end;

nonRootNodes = idxs(gW.cp(idxs)>0);
cost = numel([gW.WavBases{idxs}]) + numel([gW.WavConsts{idxs}]) + numel([gW.Centers{nonRootNodes}]); % center coincides with wavconst at root nodes

% initial cost includes only the wavelet bases, translations and net centers

if gW.opts.splitting && ~gW.opts.mergePsiCapIntoPhi % if merge, then scaling and wavelet functions are already updated, no action needed
    parentNodesRepeated = gW.cp(nonRootNodes);
    parentNodesUnique(parentNodesRepeated) = 1;
    %parentNodesUnique = find(parentNodesUnique>0);
    cost = cost - gW.opts.AmbientDimension*(sum(gW.WavDimsPsiCap(parentNodesRepeated)) - sum(gW.WavDimsPsiCap(parentNodesUnique>0))); 
    % WavDimsPsiCap stores the wavelet intersection dimension at the parent node
end

if gW.opts.addTangentialCorrections,   
    cost = cost + numel([gW.ScalFuns{nonRootNodes}]); % scaling function conincides with wavelet function at root nodes
    if gW.opts.avoidLeafnodePhi
        nonRootLeafnodes = nonRootNodes(gW.isaleaf(nonRootNodes));
        cost = cost-numel([gW.ScalFuns{nonRootLeafnodes}]);
    end
end

return;

function gW = get_subtree(gW, flags)

%
gW.flags = flags;

%%
keptNodes = find(flags==1);

inverseMap = zeros(1, length(gW.cp));
inverseMap(keptNodes) = 1:length(keptNodes);

new_cp = gW.cp(keptNodes);
innerNodes = find(new_cp>0);
new_cp(innerNodes) = inverseMap(new_cp(innerNodes));

gW.cp = new_cp;
gW.IniLabels = inverseMap(gW.IniLabels);
gW.Scales = compute_scales(gW.cp);
gW.isaleaf = leafnodes(gW.cp);
gW.LeafNodes = find(gW.isaleaf);

gW.PointsInNet = gW.PointsInNet(keptNodes);
gW.Radii = gW.Radii(keptNodes);
gW.Sizes = gW.Sizes(keptNodes);
gW.Centers = gW.Centers(keptNodes);
gW.ScalFuns = gW.ScalFuns(keptNodes);
gW.Sigmas = gW.Sigmas(keptNodes);
gW.WavBases = gW.WavBases(keptNodes);
gW.WavConsts = gW.WavConsts(keptNodes);
gW.WavSingVals = gW.WavSingVals(keptNodes);

%
if gW.opts.sparsifying && ~gW.opts.orthogonalizing
    gW.Projections = gW.Projections(keptNodes);
end

%
if gW.opts.pruning,
    gW.epsEncodingCosts = gW.epsEncodingCosts(keptNodes);
end

%
if gW.opts.splitting
    gW.opts.mergePsiCapIntoPhi = false; 
    gW.WavDimsPsiCap = zeros(1,length(gW.cp));
    nonleafNodes = find(~gW.isaleaf);
    for i = 1:length(nonleafNodes)
        gW = splitting_WaveletBases(gW, nonleafNodes(i));
    end
end

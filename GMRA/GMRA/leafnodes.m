function isaleaf = leafnodes(cp)

isaleaf = true(1, numel(cp));
isaleaf(cp(cp>0)) = false;

%leaves = find(isaleaf);
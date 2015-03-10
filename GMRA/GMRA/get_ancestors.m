function chain = get_ancestors(cp, n)

% Input:
%   cp: vector encoding the metis tree
%   n: current node
% Output:
%   chain: list of ancestors of the node n up the tree, 
%          the first element of chain is the immediate parent of n and
%          the last element is the root of the tree

chain = [];

for i = 1:length(n),
    p = cp(n(i));
    
    while p>0
        chain = [chain p];
        p = cp(p);
    end;
end;

chain = unique(chain);

return;
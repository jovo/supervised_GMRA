function chain = dpath(cp, n)

% This function finds the path down the tree cp 
% from root to the node n (including both ends).
%
% Input:
%   cp: vector encoding the metis tree
%   n: current node
% Output:
%   chain: vector of the path down the tree, 
%          the first element of chain is the root and 
%          the last element of chain is the current node n

chain = [];
while n>0
   chain = [n chain];
   n = cp(n);
end
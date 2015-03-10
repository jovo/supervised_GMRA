function j = get_scale( cp,n )

% Input:
%   cp      : vector encoding the metis tree
%   n       : current node
% Output:
%   j       : scale of node n. 1 is the coarsest scale
%

j = 1;
p = cp(n);

while p>0
   j = j+1;
   p = cp(p);
end

return;
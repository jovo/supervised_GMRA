function [offspring, offspringLeafNodes] = get_offspring(cp, node)

% This function finds all the offspring of (but not including), and all the leaf nodes it contains (including this node) of, the given node in the tree cp.
nAllNets = length(cp);

offspring = zeros(1,nAllNets);
offspringLeafNodes = zeros(1, nAllNets);

currentNodes = node;

while ~isempty(currentNodes)
    
    newNodes = zeros(1,nAllNets); % collects all the children of currentNodes
    
    for i = 1:length(currentNodes)
        children = find(cp == currentNodes(i));
        if ~isempty(children)
            newNodes(children)=1;
        else
            offspringLeafNodes(currentNodes(i)) = 1;
        end
    end
    
    currentNodes = find(newNodes>0);

    offspring(currentNodes) = 1;
    
end

offspring = find(offspring>0);

if nargin>1,
    offspringLeafNodes = find(offspringLeafNodes>0);
end
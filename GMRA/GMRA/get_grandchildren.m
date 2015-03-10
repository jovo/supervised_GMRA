function [grandchildren, leafNodesChildren, nonLeafNodesChildren]...
    = get_grandchildren(cp, node, children)
   
% grandchildren = [];
% leafNodesChildren = [];
% nonLeafNodesChildren = [];
% 
% for c = 1:length(children)
%     
%     grch = find(cp==children(c));
%     
%     if ~isempty(grch)
%         nonLeafNodesChildren = [nonLeafNodesChildren children(c)];
%         grandchildren = [grandchildren grch];
%     else
%         leafNodesChildren = [leafNodesChildren children(c)];
%     end
%     
% end

if nargin<3; children = find(cp==node); end

nChildren = numel(children);

if nChildren > 0,
    
    grandchildren = cell(1, nChildren);
    isaleaf = ones(1, nChildren);
    
    for c = 1:nChildren,
        
        grch = find(cp==children(c));
        
        if ~isempty(grch),
            grandchildren{c} = grch;
            isaleaf(c) = 0;
        end
        
    end
    
    grandchildren = [grandchildren{:}];
    
    if nargout>1,
        leafNodesChildren = children(isaleaf>0);
        nonLeafNodesChildren = children(isaleaf==0);
    end
    
else
    
    grandchildren = [];
    
    if nargout>1,
        leafNodesChildren = [];
        nonLeafNodesChildren = [];
    end
    
end

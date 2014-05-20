function treeplot(p,c,d,labels,vertexstyle,edgestyle,labels2)
%TREEPLOT Plot picture of tree.
%   TREEPLOT(p) plots a picture of a tree given a row vector of
%   parent pointers, with p(i) == 0 for a root. 
%
%   TREEPLOT(P,nodeSpec,edgeSpec) allows optional parameters nodeSpec
%   and edgeSpec to set the node or edge color, marker, and linestyle.
%   Use '' to omit one or both.
%
%   Example:
%      treeplot([2 4 2 0 6 4 6])
%   returns a complete binary tree.
%
%   See also ETREE, TREELAYOUT, ETREEPLOT.

%   Copyright 1984-2009 The MathWorks, Inc. 
%   $Revision: 5.12.4.3 $  $Date: 2009/04/21 03:26:23 $

if (nargin<4) || (isempty(labels)), labels=(1:length(p)); end;
if (nargin<5) || (isempty(vertexstyle)), vertexstyle = 'ro'; end;
if (nargin<6) || (isempty(edgestyle)), edgestyle = 'r-'; end;
if (nargin<7), labels2 = []; end;

[x,y,h]=treelayout(p);
f = find(p~=0);
pp = p(f);
X = [x(f); x(pp); NaN(size(f))];
Y = [y(f); y(pp); NaN(size(f))];
X = X(:);
Y = Y(:);

if (nargin == 1) || (isempty(c))
    n = length(p);
    if n < 500,
        plot (x, y, vertexstyle, X, Y, edgestyle);
        for k = 1:n,
            if labels(k)~=0,
                text(x(k)+0.02,y(k)-0.000,num2str(labels(k)));
            end;
            if ~isempty(labels2),
                if labels2(k)~=0,
                    text(x(k)-0.01,y(k)-0.05,num2str(labels2(k)));
                end;
            end;
        end;
    else
        plot (X, Y, edgestyle);
    end;
else
    [~, clen] = size(c);
    if nargin < 3, 
        if clen > 1, 
            d = [c(1:clen-1) '-']; 
        else
            d = 'r-';
        end;
    end;
    [~, dlen] = size(d);
    if clen>0 && dlen>0
        plot (x, y, c, X, Y, d);
    elseif clen>0,
        plot (x, y, c);
    elseif dlen>0,
        plot (X, Y, d);
    else
    end;
end;
xlabel(['height = ' int2str(h)]);
axis([0 1 0 1]);

function treeplotimages(GWT,maxscales)
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
% EXAMPLE:
% load test_images; treeplotimages(gW,8)
%

if nargin<2, maxscales = inf; end;

[x,y,h]=treelayout(GWT.cp);
f = find(GWT.cp~=0);
pp = GWT.cp(f);
X = [x(f); x(pp); repmat(NaN,size(f))];
Y = [y(f); y(pp); repmat(NaN,size(f))];
X = X(:);
Y = Y(:);

n = length(GWT.cp);
p=plot (x, y, 'ro', X, Y, 'r--');
outeraxespos = get(gca,'Position');

% compute the spacing between vertices at the same scale
for j = 0:min([maxscales,h]);
    idxs = find(GWT.Scales==j);
    maxwidth(j+1) = inf;
    for k = 2:length(idxs),
        maxwidth(j+1) = min([maxwidth,abs(x(idxs(k))-x(idxs(1)))]);
    end;
end;
maxwidth(find(maxwidth==inf)) = max(maxwidth(find(maxwidth<inf)));

for k = 1:n,
    if GWT.Scales(k)<=maxscales,
        %imageatnode=DisplayImageCollection(reshape(GWT.ScalFuns{k},[16,16,size(GWT.ScalFuns{k},2)]),2,0);
        if ~isempty(GWT.WavBases{k}),
            imageatnode=DisplayImageCollection(reshape(GWT.WavBases{k},[16,16,size(GWT.WavBases{k},2)]),2,0);
            imagesize = min([maxwidth(GWT.Scales(k))/3,0.06]);
            h1=axes('position',[x(k)*outeraxespos(3)+outeraxespos(1)-imagesize/2 y(k)*outeraxespos(4)+outeraxespos(2)-imagesize/2  imagesize imagesize],'Xtick',[],'Ytick',[],'box','on');
            imagesc(imageatnode);colormap(gray);
            set(h1,'Xtick',[],'Ytick',[]);
            %axis image;
            %pos = get(h1,'position');
            %set(h1,'position',[x(k)*outeraxespos(3)+outeraxespos(1) y(k)*outeraxespos(4)+outeraxespos(2)-imagesize/2  imagesize imagesize]);
        end;
    end;
end;
xlabel(['height = ' int2str(h)]);
%axis([0 1 0 1]);
axis tight;

return;

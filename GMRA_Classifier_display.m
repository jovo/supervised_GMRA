function GMRA_Classifier_display( MRA, MRATest )

%
% function GMRA_Classifier_display( MRA,MRATest )
%
% Basic visualization of a GMRA-based classifier tested on data.
%
% (c) Copyright Mauro Maggioni 2013
%

H = figure;
treeplot(MRA.cp, 'g.', 'c');

% treeplot is limited with control of colors, etc.
P = findobj(H, 'Color', 'c');
set(P, 'Color', [247 201 126]/255);
P2 = findobj(H, 'Color', 'g');
set(P2, 'MarkerSize', 5, 'Color', [180 180 180]/255);

% count = size(GWT.cp,2);
[x,y] = treelayout(MRA.cp);
x = x';
y = y';
hold();

% Show which nodes were used (self or children)
use_self_idx = MRA.Classifier.activenode_idxs;
plot(x(use_self_idx), y(use_self_idx), 'o', 'MarkerSize', 10, 'Color', [0.8 0.5 0.5]);

if nargin>=2,
    error_array = round(MRATest.Test.errors);
    error_strings = cellstr(num2str(error_array'));
    
    handle1 = text(x(use_self_idx),y(use_self_idx)+0.01, error_strings(use_self_idx), ...
        'VerticalAlignment','bottom','HorizontalAlignment','right','Color', [0.2 0.2 0.2]);
end;

return;
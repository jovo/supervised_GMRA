function y = f_displaylabelserrorrate( GWT, n, Data )

y = mean(GWT.Labels(GWT.PointsInNet{n}));
y = abs(y-round(y));

return;
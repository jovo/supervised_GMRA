function PlotPolyFit(X,Y,N)

if nargin<3, N=1;   end;

fit = polyfit(X,Y,N);

hold on;
p=plot(X,polyval(fit,X),'r--', 'LineWidth', 2);

legend(p,['y=',poly2str(fit,'x')],'Location','Best');

return;
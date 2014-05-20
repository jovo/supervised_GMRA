function GWT_DisplayApproxErr( GWT, Data, ErrOpts )

%
% Plots reconstruction error (w.r.t. to clean and noisy data) as a function of scale.
%
% USES:
%   GWT_ApproxError

if nargin<3,
    ErrOpts = [];
end;

J = max(GWT.Scales);
%N = size(GWT.X,1);

% error against scale
delta = zeros(1,J);
for j = 1:J
    delta(j) = mean(GWT.Radii(GWT.Scales == j));
end;

try
    err = GWT_ApproxError( GWT.X, Data.Projections, ErrOpts );
    
    figure; plot(-log10(delta), log10(err), '*', 'MarkerSize',10);
    if J>3
        PlotPolyFit( -log10(delta(2:end-1)'), log10(err(2:end-1)) );
    elseif J>1
        PlotPolyFit( -log10(delta'), log10(err) );
    end
    xlabel('scale', 'fontSize', 12); ylabel('error', 'fontSize', 12)
    grid on
    axis equal;
    title('Error against scale (in log10 scale)', 'fontSize', 12)
catch
end;

try
    if isfield(GWT.opts, 'X_clean')
        err_clean = GWT_ApproxError( GWT.opts.X_clean, Data.Projections, ErrOpts );
        
%         figure; plot(-log10(delta), log10(err_clean), '*','MarkerSize',10);
%         %PlotPolyFit( -log10(delta(2:end-1)'), log10(err_clean(2:end-1)) );
%         axis tight;
%         title 'Error against scale (log-log plot) - clean data'
        
        figure; plot(-log10(delta), log10(err), '*','MarkerSize',10);
        hold on; 
        plot(-log10(delta), log10(err_clean), 'r^','MarkerSize',10);
        axis equal;
        legend('noisy data', 'clean data')
        title('Error against scale (in log10 scale) - clean data', 'fontSize', 12)
        
    end
catch
end;

 %figure; plot(-log10(delta(2:end-1)), log10(squeeze(mean(sum((Data.TangentialCorrections(:,:,2:end-1)).^2,2),1))), '.')


return;
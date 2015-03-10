clear
%close all

%%
dataset = 'MNIST_Digits'; 
thresholds = 10.^(-3:0.1:2);

% dataset = 'croppedYaleB_Faces';
% thresholds = 10.^(-1:0.1:4);

% dataset = 'ScienceNews'; % science news data
% thresholds = 10.^(-5:0.1:0);

% dataset = 'Oscillating2DWave';
% thresholds = 10.^(-5:0.1:0);

%%
[GWTstats0, SVDstats, thresSVDstats] = GenerateRateDistortionCurve( dataset, 0, thresholds );
GWTstats1 = GenerateRateDistortionCurve( dataset, 1, thresholds );
GWTstats2 = GenerateRateDistortionCurve( dataset, 2, thresholds );

%%
figure;
hold on
plot(log10(thresholds), log10(GWTstats0.err_X), 'b', 'MarkerSize', 12, 'linewidth', 2);
plot(log10(thresholds), log10(GWTstats1.err_X), 'm', 'MarkerSize', 12, 'linewidth', 2);
plot(log10(thresholds), log10(GWTstats2.err_X), 'r', 'MarkerSize', 12, 'linewidth', 2);
plot(log10(thresholds), log10(thresSVDstats.err_X), 'k', 'MarkerSize', 12, 'linewidth', 2);
legend('GWT','OrthogonalGWT','PruningGWT', 'thresSVD')
xlabel('threshold', 'fontSize', 12)
ylabel('error', 'fontSize', 12)
grid on
box on

% overall cost
figure
title 'overall costs'
hold on
plot(GWTstats0.err_X,(GWTstats0.totalCosts), 'b', 'MarkerSize', 12, 'linewidth', 2);
plot((GWTstats1.err_X),(GWTstats1.totalCosts), 'm', 'MarkerSize', 12, 'linewidth', 2);
plot((GWTstats2.err_X),(GWTstats2.totalCosts), 'r', 'MarkerSize', 12, 'linewidth', 2);
plot((thresSVDstats.err_X),(thresSVDstats.totalCosts),'k', 'MarkerSize', 12, 'linewidth', 2);
plot((SVDstats.err_X),(SVDstats.totalCosts),'k--', 'MarkerSize', 12, 'linewidth', 2);
xlabel('error', 'fontSize', 12)
ylabel('cost', 'fontSize', 12)
legend('GWT','OrthogonalGWT','PruningGWT', 'thresSVD', 'SVD')
grid on
box on

figure
title 'overall costs'
hold on
plot(log10(GWTstats0.err_X),log10(GWTstats0.totalCosts), 'b', 'MarkerSize', 12, 'linewidth', 2);
plot(log10(GWTstats1.err_X),log10(GWTstats1.totalCosts),  'm', 'MarkerSize', 12, 'linewidth', 2);
plot(log10(GWTstats2.err_X),log10(GWTstats2.totalCosts),  'r', 'MarkerSize', 12, 'linewidth', 2);
plot(log10(thresSVDstats.err_X), log10(thresSVDstats.totalCosts),'k', 'MarkerSize', 12, 'linewidth', 2);
plot(log10(SVDstats.err_X), log10(SVDstats.totalCosts),'k--', 'MarkerSize', 12, 'linewidth', 2);
xlabel('error', 'fontSize', 12)
ylabel('cost', 'fontSize', 12)
legend('GWT','OrthogonalGWT','PruningGWT', 'thresSVD', 'SVD')
grid on
box on

% dictionary cost
figure
title 'dictionary costs'
hold on
plot(log10(GWTstats0.err_X),log10(GWTstats0.dictCosts), 'b', 'MarkerSize', 12, 'linewidth', 2);
plot(log10(GWTstats1.err_X),log10(GWTstats1.dictCosts),  'm', 'MarkerSize', 12, 'linewidth', 2);
plot(log10(GWTstats2.err_X),log10(GWTstats2.dictCosts),  'r', 'MarkerSize', 12, 'linewidth', 2);
plot( log10(thresSVDstats.err_X), log10(thresSVDstats.dictCosts),'k', 'MarkerSize', 12, 'linewidth', 2);
plot( log10(SVDstats.err_X), log10(SVDstats.dictCosts),'k--', 'MarkerSize', 12, 'linewidth', 2);
xlabel('error', 'fontSize', 12)
ylabel('cost', 'fontSize', 12)
legend('GWT','OrthogonalGWT','PruningGWT', 'thresSVD', 'SVD')
grid on
box on

% coeff cost
figure
title 'coeffs costs'
hold on
plot(log10(GWTstats0.err_X),log10(GWTstats0.coeffsCosts), 'b', 'MarkerSize', 12, 'linewidth', 2);
plot(log10(GWTstats1.err_X),log10(GWTstats1.coeffsCosts),  'm', 'MarkerSize', 12, 'linewidth', 2);
plot(log10(GWTstats2.err_X),log10(GWTstats2.coeffsCosts),  'r', 'MarkerSize', 12, 'linewidth', 2);
plot( log10(thresSVDstats.err_X), log10(thresSVDstats.coeffsCosts),'k', 'MarkerSize', 12, 'linewidth', 2);
plot( log10(SVDstats.err_X), log10(SVDstats.coeffsCosts),'k--', 'MarkerSize', 12, 'linewidth', 2);
xlabel('error', 'fontSize', 12)
ylabel('cost', 'fontSize', 12)
legend('GWT','OrthogonalGWT','PruningGWT', 'thresSVd', 'SVD')
grid on
box on





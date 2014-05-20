%
% Computational speed test for GWT
%

clear all; close all;pack;clc;

% Run full GWT on n points on the k-dimensional sphere in R^D, with increasing n and k and D, and different amounts of noise
pN = [1000,2000,4000,8000,16000,32000];
pk = [2,4,8,16,32];
pD = [100,1000];
pNoiseNorm = [0,0.25,0.5];


%% Loop through the parameters
%
%  Set GWT parameters
%
GWTopts = struct();
GWTopts.errorType = 'relative'; % or absolute
GWTopts.threshold0 = 1e-2; % threshold for choosing pca dimension at each nonleaf node
GWTopts.precision  = 1e-2; % only for leaf nodes
% The following thresholds are used in the code construct_GMRA.m
GWTopts.threshold1 = 1e-1; % threshold of singular values for determining the rank of each ( I - \Phi_{j,k} * \Phi_{j,k} ) * Phi_{j+1,k'}
GWTopts.threshold2 = 5e-2; % threshold for determining the rank of intersection of ( I - \Phi_{j,k} * \Phi_{j,k} ) * Phi_{j+1,k'}
% The following parameter .pruning determines which version of geometric wavelets to use
GWTopts.pruning = 1;
% whether to use best approximations
GWTopts.addTangentialCorrections = true;
% whether to sparsify the scaling functions and wavelet bases
GWTopts.sparsifying = false;
% whether to split the wavelet bases into a common intersection and
% children-specific parts
GWTopts.splitting = false;
% METIS parameters
GWTopts.knn = 50;
GWTopts.knnAutotune = 30;
GWTopts.smallestMetisNet = 20;

Timings.GW              = zeros(length(pN),length(pk),length(pNoiseNorm),length(pD));
Timings.nesdis          = zeros(size(Timings.GW));
Timings.graph           = zeros(size(Timings.GW));
Timings.TrainingData    = zeros(size(Timings.GW));

for n = 1:length(pN),
    fprintf('\n %d/%d\n',n,length(pN));
    for k = 1:length(pk),
        for noisenorm = 1:length(pNoiseNorm),
            for d = 1:length(pD),   
                global gW;
                
                % Data set parameters
                dataset.N = pN(n);
                dataset.k = pk(k);
                dataset.D = pD(d);
                dataset.noiseLevel = pNoiseNorm(noisenorm)/sqrt(dataset.D);                
                dataset.name = 'D-Sphere';
                GWTopts.ManifoldDimension = dataset.k;                                
                % Generate data
                X_clean = GenerateDataSets( dataset.name, struct('NumberOfPoints',dataset.N,'Dim',dataset.k,'EmbedDim',dataset.D,'NoiseType','Uniform','NoiseParam',0) );
                % Add noise
                X = X_clean + dataset.noiseLevel*random('norm', 0,1, size(X_clean)); % N by D data matrix                
                gW.X = X;
                gW.X_clean = X_clean;                
                GWTopts.verbose = 1;
                
                %% Construct geometric wavelets
                gW = GMRA(gW.X,GWTopts);
                Timings.graph(n,k,noisenorm,d)  = gW.Timing.graph;
                Timings.nesdis(n,k,noisenorm,d) = gW.Timing.nesdis;
                Timings.GW(n,k,noisenorm,d)     = gW.Timing.GW;
                
                %% Computing all wavelet coefficients
                tic
                Data = GWT_trainingData(gW, X);
                Timings.TrainingData(n,k,noisenorm,d) = toc;  
                
                clear gW
                clear global
            end;            
        end;
    end;
end;


%% Display results
lFigTitle = 'Timing1';
figure;b=bar(log10(pN),[log10(squeeze(Timings.nesdis(:,3,1,1))),log10(squeeze(Timings.nesdis(:,3,3,1))),  ...
    log10(squeeze(Timings.graph(:,1,1,1))),log10(squeeze(Timings.graph(:,1,3,1))),  ...
    log10(squeeze(Timings.GW(:,1,1,1))),log10(squeeze(Timings.GW(:,1,3,1)))]+3,'grouped');
set(gca,'YLim',[0,8]);
xlabel('log_{10}(n)');ylabel('log_{10}(T)    msec.'); title(['Computation Time for 8-D sphere embedded in 100-d',[]]);
legend({'Tree','Tree (w noise)','Graph','Graph (w noise)','Geom. Wav.','Geom. Wav. (w noise)'},'location','NorthWest');
print(gcf,'-depsc2',sprintf('Figures//%s.eps',lFigTitle));

lFigTitle = 'Timing2';
figure;b=bar(log10(pN),[log10(squeeze(Timings.nesdis(:,3,1,2))),log10(squeeze(Timings.nesdis(:,3,3,2))),  ...
    log10(squeeze(Timings.graph(:,1,1,2))),log10(squeeze(Timings.graph(:,1,3,2))),  ...
    log10(squeeze(Timings.GW(:,1,1,2))),log10(squeeze(Timings.GW(:,1,3,2)))]+3,'grouped');
set(gca,'YLim',[0,8]);
xlabel('log_{10}(n)');ylabel('log_{10}(T)    msec.'); title(['Computation Time for 8-D sphere embedded in 1000-d',[]]);
legend({'Tree','Tree (w noise)','Graph','Graph (w noise)','Geom. Wav.','Geom. Wav. (w noise)'},'location','NorthWest');
print(gcf,'-depsc2',sprintf('Figures//%s.eps',lFigTitle));

lFigTitle = 'Timing3';
figure;bar(log2(pk),[(squeeze(Timings.nesdis(end,:,1,1)))',(squeeze(Timings.nesdis(end,:,3,1)))', ...
                     log10(squeeze(Timings.graph(end,:,1,1)))',log10(squeeze(Timings.graph(end,:,3,1)))', ...
                     log10(squeeze(Timings.GW(end,:,3,1)))', log10(squeeze(Timings.GW(end,:,3,1)))']+3,'grouped');
xlabel('log_2(dim(M))');ylabel('log_{10}(T)    msec.'); title(['Computation Times for sphere of dimension dim(M)',[]]);
legend({'Tree','Tree (w noise)','Graph','Graph (w noise)','Geom. Wav.','Geom. Wav. (w noise)'},'location','NorthWest');
print(gcf,'-depsc2',sprintf('Figures//%s.eps',lFigTitle));


save TestGWTSpeed p* GWTopts Timings

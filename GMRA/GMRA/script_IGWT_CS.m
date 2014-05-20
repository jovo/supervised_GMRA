%clear all;close all;clc; clear global

%% Set parameters
pValN           = 5;
pMaxDimForNNs   = inf;
pRandomProjDim  = [1,2,4,16];       % Projection dimension by oversampling the maximum intrinsic dimension of planes
pNoise          = [0,0.05,0.1];
pSpaRSAMaxSize  = 50;

DataSet(1).name = 'SwissRoll';
DataSet(1).opts = struct('NumberOfPoints',4000,'EmbedDim',100);
DataSet(2).name = 'D-Sphere';
DataSet(2).opts = struct('NumberOfPoints',40000,'Dim',8,'EmbedDim',100);
DataSet(3).name = 'BMark_MNIST';
DataSet(3).opts = struct('NumberOfPoints',5000,'BMarkUnitBall',1,'MnistOpts',struct('Sampling', 'RandN', 'QueryDigits',[1], 'ReturnForm', 'vector'));
DataSet(4).name = 'BMark_MNIST';
DataSet(4).opts = struct('NumberOfPoints',5000,'BMarkUnitBall',1,'MnistOpts',struct('Sampling', 'RandN', 'QueryDigits',[1,3,5,7], 'ReturnForm', 'vector'));
DataSet(5).name = 'ScienceNews';
DataSet(5).opts = struct('ReturnForm', 'vector');


% Set parameters for GWT
GWTopts{1} = struct('GWTversion',0);
GWTopts{1}.ManifoldDimension = 2;
GWTopts{1}.threshold1 = 1e-3;
GWTopts{1}.threshold2 = .1;
GWTopts{1}.addTangentialCorrections = false;
GWTopts{1}.sparsifying = false;
GWTopts{1}.splitting = false;
GWTopts{1}.knn = 30;
GWTopts{1}.knnAutotune = 20;
GWTopts{1}.smallestMetisNet = 30;
GWTopts{1}.verbose = 1;
GWTopts{1}.shrinkage = 'hard';
GWTopts{1}.avoidLeafnodePhi = false;
GWTopts{1}.mergePsiCapIntoPhi  = true;
GWTopts{1}.coeffs_threshold = 0;
GWTopts{1}.errorType = 'relative';
GWTopts{1}.threshold0 = 0.5;
GWTopts{1}.precision  = 1e-4;
GWTopts{2}=GWTopts{1};
GWTopts{2}.ManifoldDimension = DataSet(2).opts.Dim;
GWTopts{2}.precision  = 1e-4;
GWTopts{3}=GWTopts{1};
GWTopts{3}.ManifoldDimension = 0;
GWTopts{3}.precision  = 1e-2;
GWTopts{4}=GWTopts{3};
GWTopts{5}=GWTopts{4};

%% Go parallel
if matlabpool('size')==0,
    matlabpool
end;

%% Create the GWT and SVD models (this is fast enough)
fprintf('\n Generating GMRA models...');

lErrAbs         = cell(1,length(DataSet));
lErrRel         = cell(1,length(DataSet));
lErrApprox      = cell(1,length(DataSet));
NumberOfPlanes  = cell(1,length(DataSet));
DataGWT         = cell(1,length(DataSet));

X_train = cell(1,length(DataSet));
gMRA    = cell(1,length(DataSet));
DataGWT = cell(1,length(DataSet));
RandomProjDim = cell(1,length(DataSet));

timings = struct;

x_SpaRSA_m = [];

%% Go through data sets
for i = 1:length(DataSet),                                                                                  % Go through data sets (i)
    %% Generate the training data set and the validation data set
    fprintf('\n\n---- Data set %d --------',i);
    X_train{i} = GenerateDataSets( DataSet(i).name, DataSet(i).opts );
    
    %% Compute GWT and the transform of the data
    fprintf('\n Computing GWT...');
    gMRA{i}          = GMRA(X_train{i},GWTopts{i});
    DataGWT{i}       = FGWT(gMRA{i},X_train{i});    
        
    %% Get new points and randomly project them
    nScales                 = max(gMRA{i}.Scales);
    lErrAbs{i}              = zeros(nScales,pValN,length(pRandomProjDim),length(pNoise));
    lErrApprox{i}           = zeros(nScales,pValN,length(pRandomProjDim),length(pNoise));
    lErrApproxCS{i}         = zeros(nScales,pValN,length(pRandomProjDim),length(pNoise));
    lErrRel{i}              = zeros(nScales,pValN,length(pRandomProjDim),length(pNoise));
    lErrRelApprox{i}        = zeros(nScales,pValN,length(pRandomProjDim),length(pNoise));
    lErrRelApproxCS{i}      = zeros(nScales,pValN,length(pRandomProjDim),length(pNoise));
    lErrAbsInf{i}           = zeros(nScales,pValN,length(pRandomProjDim),length(pNoise));
    lErrApproxInf{i}        = zeros(nScales,pValN,length(pRandomProjDim),length(pNoise));
    lErrApproxCSInf{i}      = zeros(nScales,pValN,length(pRandomProjDim),length(pNoise));
    lErrRelInf{i}           = zeros(nScales,pValN,length(pRandomProjDim),length(pNoise));
    lErrRelApproxInf{i}     = zeros(nScales,pValN,length(pRandomProjDim),length(pNoise));
    lErrRelApproxCSInf{i}   = zeros(nScales,pValN,length(pRandomProjDim),length(pNoise));
    lErrRelApproxSpaRSA{i}  = zeros(nScales,pValN,length(pRandomProjDim),length(pNoise));
    
    for p = 1:pValN,                                                                                        % Go through validation runs (p)
        for n = 1:length(pNoise),
            fprintf('\n Validation cycle %d/%d...dim=',p,pValN);
            X_val = GenerateDataSets( DataSet(i).name, DataSet(i).opts );
            X_val_n = X_val+pNoise(n)*1/sqrt(size(X_val,2))*randn(size(X_val));
            DataValGWT                          = FGWT(gMRA{i}, X_val);
            DataValIGWT                         = IGWT(gMRA{i}, DataValGWT);
            for m = length(pRandomProjDim):-1:1,                                                                   % Go through projection dimensions (m)
                fprintf('%d,',m);
                % Do CS inversion on the new points
                %hatx = zeros(size(X_val));
                for j = nScales:-1:1,                                                                              % Go through number of scales (j)
                    lPartition = get_partition_at_scale(gMRA{i},j);
                    NumberOfPlanes{i}(j,p,m) = length(lPartition);
                    tmp = [];
                    for u = 1:length(lPartition),
                        tmp = [tmp, size(gMRA{i}.ScalFuns{lPartition(u)},2)];
                    end;
                    RandomProjDim{i}(j,p,m,n)      = round(pRandomProjDim(m)*mean(tmp));                           % Projection dimension by oversampling the maximum intrinsic dimension of planes
                    [X_val_proj, Proj]             = RandomProject( X_val_n, RandomProjDim{i}(j,p,m,n) );
                    
                    % Run our inversion algorithm
                    timings.IGWT_CS(i,j,p,m,n)     = cputime;
                    [hatx,~,l0_norm,l1_norm]       = IGWT_CS( gMRA{i}, X_val_proj, Proj, j );
                    timings.IGWT_CS(i,j,p,m,n)     = cputime-timings.IGWT_CS(i,j,p,m,n);
                    timings.IGWT_CS(i,j,p,m,n)     = timings.IGWT_CS(i,j,p,m,n)/size(X_val_proj,2);
                    
                    % Run SpaRSA
                    D             = [[gMRA{i}.ScalFuns{lPartition}],[gMRA{i}.Centers{lPartition}]];
                    ProjD         = Proj*D;
                    ProjDt        = ProjD';
                    hR            = @(x) ProjD*x;
                    hRt           = @(x) ProjDt*x;                    
                    tau_max       = max(max(abs(hRt(X_val_proj))));
                    tau           = 0.001 * tau_max;

                    x_SpaRSA      = zeros(size(D,2),min([pSpaRSAMaxSize,size(hatx,2)]));
                    rnd_idxs      = randperm(size(hatx,2));
                    rnd_idxs      = rnd_idxs(1:size(x_SpaRSA,2));
                    timings.SpaRSA_forward(i,j,p,m,n) = cputime;
                    parfor b = 1:size(x_SpaRSA,2),
                        x_SpaRSA(:,b)= SpaRSA(X_val_proj(:,rnd_idxs(b)),hR,tau,...
                            'Monotone',1,...
                            'Debias',0,...
                            'AT',hRt,...`
                            'Initialization',0,...
                            'StopCriterion',4,...
                            'ToleranceA',1e-2,...
                            'MaxiterA',1000,...
                            'Verbose',0,...
                            'Continuation',1);
                    end;
                    timings.SpaRSA_forward(i,j,p,m,n) = cputime-timings.SpaRSA_forward(i,j,p,m,n);
                    timings.SpaRSA_forward(i,j,p,m,n) = timings.SpaRSA_forward(i,j,p,m,n)/size(x_SpaRSA,2);
                                        
                    lErrAbs{i}(j,p,m,n)            = mean(sum((squeeze(DataValIGWT(:,:,j))-hatx).^2,1),2);                                               % MSE between IGWT_CS and IGWT(j)
                    lErrApprox{i}(j,p,m,n)         = mean(sum((squeeze(DataValIGWT(:,:,j))-X_val(:,:)).^2,1),2);                                         % MSE between IGWT and IGWT(J)
                    lErrApproxCS{i}(j,p,m,n)       = mean(sum(squeeze(hatx-X_val).^2,1),2);                                                              % MSE between IGWT_CS and IGWT(J)
                    lErrRel{i}(j,p,m,n)            = mean(sum((squeeze(DataValIGWT(:,:,j))-hatx),1).^2./(sum(squeeze(DataValIGWT(:,:,j)).^2,1)),2);      % Relative MSE between IGWT_CS and IGWT(j)
                    lErrRelApprox{i}(j,p,m,n)      = mean(sum((squeeze(DataValIGWT(:,:,j))-X_val),1).^2./(sum(X_val.^2)),2);                             % Relative MSE between IGWT and IGWT(J)
                    lErrRelApproxCS{i}(j,p,m,n)    = mean(sum((hatx-X_val).^2,1)./sum(X_val.^2,1),2);                                                    % Relative MSE between IGWT_CS and IGWT(J)
                    lErrRelApproxSpaRSA{i}(j,p,m,n)= mean(sum((D*x_SpaRSA-X_val(:,rnd_idxs)).^2,1)./sum(X_val(:,rnd_idxs).^2,1),2);                   % Relative MSE between SpaRSA and IGWT(J)
                    lErrAbsInf{i}(j,p,m,n)         = max(sum((squeeze(DataValIGWT(:,:,j))-hatx).^2,1),[],2);                                             % Sup distance between IGWT_CS and IGWT(j)
                    lErrApproxInf{i}(j,p,m,n)      = max(sum((squeeze(DataValIGWT(:,:,j))-X_val(:,:)).^2,1),[],2);                                       % Sup distance between IGWT and IGWT(J)
                    lErrApproxCSInf{i}(j,p,m,n)    = max(sum(squeeze(hatx-X_val).^2,1),[],2);                                                            % Sup distance between IGWT_CS and IGWT(J)
                    lErrRelInf{i}(j,p,m,n)         = max(sum((squeeze(DataValIGWT(:,:,j))-hatx),1).^2./(sum(squeeze(DataValIGWT(:,:,j)).^2,1)),[],2);    % Relative sup distance between IGWT_CS and IGWT(j)
                    lErrRelApproxInf{i}(j,p,m,n)   = max(sum((squeeze(DataValIGWT(:,:,j))-X_val),1).^2./(sum(X_val.^2)),[],2);                           % Relative sup distance between IGWT and IGWT(J)
                    lErrRelApproxCSInf{i}(j,p,m,n) = max(sum((hatx-X_val).^2)./(sum(X_val.^2)),[],2);                                                    % Relative sup distance between IGWT_CS and IGWT(J)
                    lErrRelApproxSpaRSAInf{i}(j,p,m,n) = max(sum((D*x_SpaRSA-X_val(:,rnd_idxs)).^2)./(sum(X_val(:,rnd_idxs).^2)),[],2);               % Relative sup distance between SpaRSA and IGWT(J)
                    lSparsityLevelSpaRSA0{i}(j,p,m,n)  = mean(sum(abs(x_SpaRSA)>0,1));
                    lSparsityLevelSpaRSA1{i}(j,p,m,n)  = mean(sum(abs(x_SpaRSA),1));
                    lSparsityLevel0{i}(j,p,m,n)  = l0_norm;
                    lSparsityLevel1{i}(j,p,m,n)  = l1_norm;
                end;
            end;
            fprintf('\b');
        end;
    end;
    
    fprintf('\n---- done data set %d --------',i);
end;


fprintf('\n');

return;







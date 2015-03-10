%clear
close all
clc

%%
I0 = imread('barbara.jpg');
I0 = double(I0); 
%I0 = I0(146:end-95,1:480, 1);
figure; imagesc(I0); colormap gray; title 'original image'

% specify patch parameters 
basePatchSize = 16;
spacing = basePatchSize/4; % number of line pixels between consecutive patches;
% when spacing=1, all the image patches are used; on the other hand, when
% spacing = basePatchSize, nonoverlapping patches are used.
overlapping = basePatchSize-spacing; % a number (between 0 and basePatchSize-1) of pixel lines in overlapping. 

patchSize = 64; % patch size at current scale
I = I0; % initialization, stores finer details left in the image
while patchSize>basePatchSize
    
    % averaging
    h = fspecial('average', patchSize/4);
    I_filtered = imfilter(I, h);
    
    % subsample to base patch size
    step = patchSize/basePatchSize;
    I_filtered = I_filtered(1:step:end, 1:step:end);

    % extract patches
    [P,dummy,P_xy] = get_patches(I_filtered, struct('basePatchSize',basePatchSize,'spacing',overlapping,'normalizing',0));
    P = P';
    figure; do_plot_data(P', [], struct('view', 'pca')); title 'patch cloud'
    
    % apply GWRA to learn a dictionary
    [gW1, Data1] = applyGWRA2ImagePatches(P);
    DisplayImageCollection(reshape([gW1.WavBases{:}], basePatchSize,basePatchSize,[]), 50, 1, 1);
    title 'GWRA dictionary'
    
    figure; % plot the image at current coarse scale and GWRA reconstruction
    subplot(1,2,1); imagesc(I_filtered); colormap gray; title(['current coarse scale, patch size = ' int2str(patchSize)])
    Ip1 = combine_patches(Data1.Projections(:,:,end),size(I_filtered),overlapping,P_xy); 
    subplot(1,2,2); imagesc(Ip1); colormap gray; title('GWRA reconstruction');
    
    % subtract the coarse component to obtain finer details
    I = I - kron(I_filtered, ones(step, step));
    figure; imagesc(I); colormap gray; title(['details left when patch size = ' int2str(patchSize)])
    
    % for next iteration
    patchSize = patchSize/2;
    
    pause
    close all

end

%% finest scale 
% current patch Size = basePatchSize, step = 1 (no subsampling)
P = get_patches(I, struct('basePatchSize',basePatchSize,'spacing',overlapping,'normalizing',1))';
figure; do_plot_data(P, [], struct('view', 'pca')); title 'finest patches'
    
[gW, Data] = applyGWRA2ImagePatches(P);
DisplayImageCollection(reshape([gW.WavBases{:}], basePatchSize,basePatchSize,[]), 50, 1, 1);
title 'GWRA dictionary obtained at finest scale'

figure; % plot the image at the finest scale and the GWRA reconstruction
subplot(1,2,1); imagesc(I); colormap gray; title('finest scale')
Ip = combine_patches(Data.Projections(:,:,end), size(I), overlapping); 
subplot(1,2,2); imagesc(Ip); colormap gray; title('GWRA reconstruction');

%figure; imagesc(Ip+kron(Ip1, ones(2,2))); colormap gray
 
return;

%% to compare with KSVD when applied directly to the image patches with basePatchSize
params.data = I0';
params.Edata = 0.1;
params.dictsize = 100;
params.iternum = 30;
params.memusage = 'high';

[Dksvd,g,err] = ksvd(params,'');
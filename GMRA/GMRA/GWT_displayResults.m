function GWT_displayResults(gW, Data, imgOpts)

[D, N] = size(gW.X);

if nargin<3,
    imgOpts = struct();
end

if ~isfield(imgOpts, 'imageData')
    imgOpts.imageData = false;
end

if ~isfield(imgOpts, 'isCompressed')
    imgOpts.isCompressed = false;
end

%% Display the coefficents
GWT_DisplayCoeffs( gW, Data );

%% Plot approximation error
GWT_DisplayApproxErr( gW, Data, struct('norm', 2, 'relative', true) );

% %% Plot the reconstructed manifold at all scales
% J = max(gW.Scales);
% err = GWT_ApproxError( gW.X, Data.Projections, struct('norm', 2, 'relative', true) );
% 
% figure;
% for j = 1:J,
%     subplot(2, ceil(J/2), j)
%     do_plot_data(Data.Projections(:,:,j)', [], struct('view', 'pca'));
%     title(['scale = ' int2str(j) ', error = ' num2str(err(j))])
% end
% 
% % for j = 1:J,
% %     figure;
% %     do_plot_data(Data.Projections(:,:,j)', [], struct('view', 'first3'));
% %     title(['scale = ' int2str(j) ', error = ' num2str(err(j))] , 'fontSize', 12)
% % end

%% if image data
if imgOpts.imageData,
    
    if isfield(imgOpts, 'sampleImage')
        i = imgOpts.sampleImage;
    else
        i = randsample(N,1);
    end
    
    leafNode = gW.IniLabels(i);
    %leafNode = find_nearest_leaf_node(gW,gW.X(i,:)); %
    j_max = gW.Scales(leafNode);
    
    if isfield(imgOpts, 'X0') && imgOpts.isCompressed
        nSubfigs = j_max+2; % number of subfigures to display
    else
        nSubfigs = j_max+1;
    end
    nFigsPerRow = ceil(nSubfigs/3);
    
    figure;
    for j = 1:j_max
        if imgOpts.isCompressed
            X_approx = imgOpts.U(:,1:D)*Data.Projections(:,i,j)+imgOpts.cm;
        else
            X_approx = Data.Projections(:,i,j);
        end
        subplot(3,nFigsPerRow,j); imagesc(reshape(X_approx, imgOpts.imR, imgOpts.imC))
        set(gca, 'xTick', [], 'yTick', [])
        title(num2str(j)); colormap gray
    end
    
    %% original but projected
    if imgOpts.isCompressed
        j = j+1;
        X_proj = imgOpts.U(:,1:D)*gW.X(:,i)+imgOpts.cm;
        subplot(3,nFigsPerRow,j); imagesc(reshape(X_proj, imgOpts.imR, imgOpts.imC))
        title 'projection'
        set(gca, 'xTick', [], 'yTick', [])
        colormap gray
        
        %% original
        if isfield(imgOpts, 'X0')
            j = j+1;
            X_orig = imgOpts.X0(:,i);
            subplot(3,nFigsPerRow,j); imagesc(reshape(X_orig, imgOpts.imR, imgOpts.imC))
            title 'original'
            set(gca, 'xTick', [], 'yTick', [])
            colormap gray
        end
        
    else % not compressed, then X is the original, X0 is not relevant
        j = j+1;
        X_orig = gW.X(:,i);
        subplot(3,nFigsPerRow,j); imagesc(reshape(X_orig, imgOpts.imR, imgOpts.imC))
        title 'original'
        set(gca, 'xTick', [], 'yTick', [])
        colormap gray
    end
    
    %%
    chain = dpath(gW.cp, leafNode);
    
    figure;
    for j = 1:j_max
        if ~isempty(gW.WavBases{chain(j)})
            wavDict_j = gW.WavBases{chain(j)};
            if imgOpts.isCompressed
                wavDict_j = imgOpts.U(:,1:D)*wavDict_j;
            end
            appWavDict_j = [wavDict_j; min(min(wavDict_j))*ones(imgOpts.imR, size(wavDict_j,2))];
            matWavDict_j = reshape(appWavDict_j, imgOpts.imR, []);
            matWavDict_j = matWavDict_j(:,1:end-1);
            subplot(3,ceil(length(chain)/3),j);
            imagesc(matWavDict_j);
            title(num2str(j)); colormap gray
            set(gca, 'xTick', [], 'yTick', [])
            %colormap(map2);
            %balanceColor;
        end
    end
    
end

return;
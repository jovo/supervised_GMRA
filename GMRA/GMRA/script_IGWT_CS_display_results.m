%% Display results from script_IGWT_CS

clear lErr lErrStd lErrInf lErrStdInf

lLineWidth_A = 1.5;
lLineWidth_SpaRSA = 0.5;

for i = 1:length(gMRA),
    for n = 1:length(pNoise),
        clear Radii
        for j = max(gMRA{i}.Scales):-1:1,
            Radii(j) = min(gMRA{i}.Radii(gMRA{i}.Scales==j));
        end;
        
        % Plot error statistics
        figure;
        clear PlotHandles legendStrings handles
        for m = 1:length(pRandomProjDim)-1,
            for j = 1:size(lErrRelApprox{i},1),
                lErr{i}(j,m,n)                      = mean(lErrRelApproxCS{i}(j,:,m+1,n),2);
                lErrStd{i}(j,m,n)                   = std(lErrRelApproxCS{i}(j,:,m+1,n),0,2);
                lErrInf{i}(j,m,n)                   = mean(lErrRelApproxCSInf{i}(j,:,m+1,n),2);
                lErrStdInf{i}(j,m,n)                = std(lErrRelApproxCSInf{i}(j,:,m+1,n),0,2);
                lErrSpaRSA{i}(j,m,n)                = mean(lErrRelApproxSpaRSA{i}(j,:,m+1,n),2);
                lTimings{i}(j,m,n)                  = mean(timings.IGWT_CS(i,j,:,m+1,n));
                lTimingsSpaRSA{i}(j,m,n)            = mean(timings.SpaRSA_forward(i,j,:,m+1,n));
                localSparsityLevel0{i}(j,m,n)       = mean(lSparsityLevel0{i}(j,:,m+1,n));
                localSparsityLevelSpaRSA0{i}(j,m,n) = mean(lSparsityLevelSpaRSA0{i}(j,:,m+1,n));
                localSparsityLevel1{i}(j,m,n)       = mean(lSparsityLevel1{i}(j,:,m+1,n));
                localSparsityLevelSpaRSA1{i}(j,m,n) = mean(lSparsityLevelSpaRSA1{i}(j,:,m+1,n));
            end;
            legendStrings{m} = sprintf('relMSE, m=%d',pRandomProjDim(m+1));
            legendStrings{m+length(pRandomProjDim)-2+1} = sprintf('relMSE-SpaRSA, m=%d',pRandomProjDim(m+1));
            legendStringssparsity0{m} = sprintf('l_0, m=%d',pRandomProjDim(m+1));
            legendStringssparsity0{m+length(pRandomProjDim)-2+1} = sprintf('l_0-SpaRSA, m=%d',pRandomProjDim(m+1));
            legendStringssparsity1{m} = sprintf('l_1, m=%d',pRandomProjDim(m+1));
            legendStringssparsity1{m+length(pRandomProjDim)-2+1} = sprintf('l_1-SpaRSA, m=%d',pRandomProjDim(m+1));
        end;
        handles  = plot(log10(lErr{i}(1:end-1,:,n)),'-');set(handles,'LineWidth',lLineWidth_A);hold on;
        handles2  = plot(log10(lErrSpaRSA{i}(1:end-1,:,n)),':');set(handles2,'LineWidth',lLineWidth_SpaRSA);hold on;            %        handles2 = plot(log10(lErrInf{i}(1:end-1,:,n)),'--');set(handles2,'LineWidth',1.3);hold on;
%         p=plot(log10(squeeze(mean(lErr{i}(1:end-1,:,n)+lErrStd{i}(1:end-1,:,n),2))),'-.');set(p,'LineWidth',lLineWidth_A/2);hold on;
%         tmp = lErr{i}(1:end-1,:,n)-lErrStd{i}(1:end-1,:,n);
%         if min(tmp)>0,
%             plot(log10(tmp),'-.');
%         end;                                                                                                                    %        plot(log10(lErrInf{i}(1:end-1,:,n)+lErrStdInf{i}(1:end-1,:,n)),':');
        p=plot(log10(squeeze(lErrRelApprox{i}(1:end-1,1,1,n))),'k--');set(p,'LineWidth',lLineWidth_A);hold on;                  %        p2=plot(log10(squeeze(lErrRelApproxInf{i}(1:end-1,1,1,n))),'k-.');set(p,'LineWidth',1.5);
        legend([handles;handles2],legendStrings);
        figname=sprintf('%s with %.2f noise: approximation error',DataSet(i).name,pNoise(n));
        
        xlabel('j');
        ylabel('relMSE');
        
        figfilename = sprintf('%s%d with %.2f noise',DataSet(i).name,i,pNoise(n));
        figfilename(strfind(figfilename,' '))=[];
        print('-depsc2',['Figures/' figfilename '.eps']);
        saveas(gcf, ['Figures/' figfilename  '.fig'], 'fig');
        title(figname);
        

        %% Plot sparsity statistics: l_0
        figure;
        handles  = plot(log10(localSparsityLevel0{i}(1:end-1,:,n)),'-');set(handles,'LineWidth',lLineWidth_A);hold on;
        handles2  = plot(log10(localSparsityLevelSpaRSA0{i}(1:end-1,:,n)),':');set(handles2,'LineWidth',lLineWidth_SpaRSA);hold on;
        legend([handles;handles2],legendStringssparsity0);
        figname=sprintf('%s with %.2f noise: sparsity',DataSet(i).name,pNoise(n));
        
        xlabel('j');
        ylabel('relMSE');
        
        figfilename = sprintf('%s%d l_0 sparsity with %.2f noise',DataSet(i).name,i,pNoise(n));
        figfilename(strfind(figfilename,' '))=[];
        print('-depsc2',['Figures/' figfilename '.eps']);
        saveas(gcf, ['Figures/' figfilename  '.fig'], 'fig');
        title(figname);
        

        %% Plot sparsity statistics: l_1
        figure;
        handles  = plot(log10(localSparsityLevel1{i}(1:end-1,:,n)),'-');set(handles,'LineWidth',lLineWidth_A);hold on;
        handles2  = plot(log10(localSparsityLevelSpaRSA1{i}(1:end-1,:,n)),':');set(handles2,'LineWidth',lLineWidth_SpaRSA);hold on;
        legend([handles;handles2],legendStringssparsity1);
        figname=sprintf('%s with %.2f noise: sparsity',DataSet(i).name,pNoise(n));
        
        xlabel('j');
        ylabel('relMSE');
        
        figfilename = sprintf('%s%d l_1 sparsity with %.2f noise',DataSet(i).name,i,pNoise(n));
        figfilename(strfind(figfilename,' '))=[];
        print('-depsc2',['Figures/' figfilename '.eps']);
        saveas(gcf, ['Figures/' figfilename  '.fig'], 'fig');
        title(figname);

        
        %% Plot timing statistics
        figure;
        handles     = plot(log10(10^3*lTimings{i}(1:end-1,:,n)),'-');set(handles,'LineWidth',lLineWidth_A);hold on;
        handles2    = plot(log10(10^3*lTimingsSpaRSA{i}(1:end-1,:,n)),':');set(handles2,'LineWidth',lLineWidth_SpaRSA);
        
        legendStrings = {};
        for k = 1:length(handles),
            legendStrings{k} = 'A';
        end;
        for k = 1:length(handles2),
            legendStrings{k+length(handles)} = 'SpaRSA';
        end;
        legend([handles;handles2],legendStrings);

        xlabel('j');
        ylabel('msec.');

        figfilename = sprintf('Timing of %s%d with %.2f noise',DataSet(i).name,i,pNoise(n));
        figfilename(strfind(figfilename,' '))=[];
        print('-depsc2',['Figures/' figfilename '.eps']);
        saveas(gcf, ['Figures/' figfilename  '.fig'], 'fig');
        
        title(figname);

    end;
end;

fprintf('\n');
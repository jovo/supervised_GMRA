clear all;
close all;
clc

%70K: 60K + 10K (Hard Binary Classification Problem)

ACC_GMRALDA_70K =[0.9678 0.9714 0.9678 0.9665 0.9693];
ACC_LDA_70K  = [0.8627 0.8627 0.8627 0.8627 0.8627];


%5K: Train2.5K Test2.5K (Easy Binary)

ACC_GMRALDA_5K =[ 0.9948 0.9980	0.9972	0.9984	0.9968];
ACC_LDA_5K =[0.9852 0.9836	0.9884  0.9924  0.9872];

Y1(1) = mean(ACC_GMRALDA_5K);
Y2(1) = mean(ACC_LDA_5K);

%1.6K: Train0.8K Test0.8K (Easy Binary)

ACC_GMRALDA_1600 =[ 0.9975 1	0.9975	0.9975	0.9988 ];
ACC_LDA_1600 =	 [0.9625 0.9762	0.9675  0.9575  0.9663 ];
Y1(2) = mean(ACC_GMRALDA_1600);
Y2(2) = mean(ACC_LDA_1600);

%1.4K: Train0.7K Test0.7K (Easy Binary)

ACC_GMRALDA_1400 = [ 0.9971 0.9971 0.9943 0.9986 0.9986];
ACC_LDA_1400 = [0.9557 0.9600 0.9429 0.9600 0.9600];
Y1(3) = mean(ACC_GMRALDA_1400);
Y2(3) = mean(ACC_LDA_1400);

%1.2K: Train0.6K Test0.6K (Easy Binary)

ACC_GMRALDA_1200 = [0.9917 1 0.9950	0.9933 0.9983];
ACC_LDA_1200 = [0.9133 0.9133 0.9600 0.900 0.9317];
Y1(4) = mean(ACC_GMRALDA_1200);
Y2(4) = mean(ACC_LDA_1200);

%1.0K: Train0.5K Test0.5K (Easy Binary)

ACC_GMRALDA_1000 = [ 0.9960	0.9940 0.9980 0.9940 0.9880];
ACC_LDA_1000 = [0.8720  0.8120 0.8960 0.8300 0.8480 ];
Y1(5) = mean(ACC_GMRALDA_1000);
Y2(5) = mean(ACC_LDA_1000);

%0.8K: Train0.4K Test0.4K (Easy Binary)

ACC_GMRALDA_800 = [ 1	 0.9975 0.9975 1      0.9975	];
ACC_LDA_800 = [ 0.7875	 0.8075 0.7725 0.8525 0.8300	];
Y1(6) = mean(ACC_GMRALDA_800);
Y2(6) = mean(ACC_LDA_800);

%0.6K: Train0.3K Test0.3K (Easy Binary)

ACC_GMRALDA_600 = [ 0.9867	0.9967 1 0.9900  0.9933];
ACC_LDA_600 = [ 0.9633  0.9633 0.9167 0.9400 0.9633 ];
Y1(7) = mean(ACC_GMRALDA_600);
Y2(7) = mean(ACC_LDA_600);

%0.4K: Train0.2K Test0.2K (Easy Binary)

ACC_GMRALDA_400 = [ 0.9950	 0.9950 1      1      1];
ACC_LDA_400 = [	 0.9850	 0.9850 0.9750 0.9850 0.9700 ];
Y1(8) = mean(ACC_GMRALDA_400);
Y2(8) = mean(ACC_LDA_400);

% Plot of the mean of the performance
figure;
X = [5000 1600 1400 1200 1000 800 600 400];
plot(X, Y1, 'rx-');
hold on;
plot(X, Y2, 'bo-'); 
hold on;
plot(3000, mean(ACC_GMRALDA_70K), 'r^');
hold on;
plot(3000, ACC_LDA_70K, 'b^');
title('Mean Accuracy of GMRALDA vs. SimpleLDA for MNIST Easy Binary Classification', 'FontSize', 22);
legend('GMRALDA', 'SimpleLDA', 'location', 'SouthEast');
xlabel('Number of Data Points (Train+Test)', 'FontSize', 20)
ylabel('Mean Performance', 'FontSize', 20)
% text(4000, 0.9, 'TrainDataSize:TestDataSize = 1:1')
% text(4000, 0.88, 'The triangle marker represents the performance on the hard binary classification with 60K train data and 10K test data');

% Boxplot
% X = [ACC_GMRALDA_70K'; ACC_GMRALDA_5K'; ACC_GMRALDA_1600'; ACC_GMRALDA_800'; ACC_GMRALDA_400' ];
% Y = [ACC_LDA_70K'; ACC_LDA_5K'; ACC_LDA_1600'; ACC_LDA_800'; ACC_LDA_400' ];
% ACTid = [1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4 5 5 5 5 5]';
% 
% xylabel = repmat('GL',25,1);
% boxplot([X; Y], {repmat(ACTid,2,1), xylabel(:)} ,'factorgap',10)

% Boxplot example 
% Y     = rand(1000,1);
% X     = Y-rand(1000,1);
% ACTid = randi(6,1000,1);
% 
% xylabel = repmat('xy',1000,1);
% boxplot([X; Y], {repmat(ACTid,2,1), xylabel(:)} ,'factorgap',10)
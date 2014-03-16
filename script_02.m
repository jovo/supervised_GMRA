% Load spectra and labels
load spectral_dictionary_material_names_20130419_WolterData.mat
X = spectral_dictionary;

% Set GMRA options
GMRAopts.threshold0         = 0.5;
GMRAopts.threshold1         = 1e-5;
GMRAopts.threshold2         = 1e-2;
GMRAopts.knn                = 5;
GMRAopts.knnAutotune        = 2;
GMRAopts.smallestMetisNet   = 5;
GMRAopts.verbose            = 1;
GMRAopts.GWTversion         = 0;


% Set up labels
[UniqueLabels,~,Labels] = unique(material_names_classes(:,2));
Labels = Labels';

% Call GMRA_LDA
[GMRA_Classifier, GMRA] = GMRA_LDA( X, Labels, struct('GMRAopts',GMRAopts) );


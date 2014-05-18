function [MRA,ClassifierResults] = GMRAClassifyAndTest( X, trainflag, Y )

% Train classifier
Priors = hist(Y(trainflag==1),unique(Y)); Priors = Priors/sum(Priors);
%Priors = [0.9,0.1];

MRA = GMRA_Classifier( X, trainflag, Y, struct('Priors',Priors,'Classifier',@LDA_traintest) );

% Test classifier
ClassifierResults = GMRA_Classifier_test( MRA, X, trainflag, Y );
fprintf('\b\b\t done.');

% Display results
fprintf('\n Error (test) : %f\%', sum(ClassifierResults.Test.errors)/sum(trainflag==0) );

fprintf('\n Confusion matrix: \n');
confusionmat(Y(trainflag==0),ClassifierResults.Test.Labels)
fprintf('\n');

return;

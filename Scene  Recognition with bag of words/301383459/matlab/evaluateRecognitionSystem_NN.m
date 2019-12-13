fprintf('Running NN \n');
load('../data/traintest.mat');
load('visionHarris.mat');
ImS = length(test_imagenames);
TrS = length(train_imagenames);
K = 100;

HDc = zeros(TrS, 1); 
HDe = zeros(TrS, 1); 

Hcorrect_c = 0;
Hcorrect_e = 0;

HCMe = zeros(8, 8);
HCMc = zeros(8, 8);

fprintf('Running NN for Harris \n');
%find closest neighbor for each image
for i=1:ImS
    I = imread(strcat('../data/', test_imagenames{i}));
    wordMap = getVisualWords(I, filterBank, dictionary');
    h1 = getImageFeatures(wordMap, K);
    for s=1:TrS
        h2 = trainFeatures(s, :);
        HDc(s) = getImageDistance(h1, h2, 'chi2');
        HDe(s) = getImageDistance(h1, h2, 'euclidean');     
    end
    [~, minHDc] = min(HDc);
    [~, minHDe] = min(HDe);
    label_c = trainLabels(minHDc);
    label_e = trainLabels(minHDe);
    
    realL = test_labels(i);
    
    %increment number correct
    if (label_c == realL)
        Hcorrect_c = Hcorrect_c + 1;
    end
    
    if (label_e == realL)
        Hcorrect_e = Hcorrect_e + 1;
    end
    
    %filling in confusion matrix   
    HCMc(realL, label_c) = HCMc(realL, label_c) + 1;
    HCMe(realL, label_e) = HCMc(realL, label_e) + 1;
end

RDc = zeros(TrS, 1); 
RDe = zeros(TrS, 1);

Rcorrect_c = 0;
Rcorrect_e = 0;

RCMe = zeros(8, 8);
RCMc = zeros(8, 8);

load('visionRandom.mat');

fprintf('Running NN for Random \n');
%find closest neighbor for each image
for i=1:ImS
    I = imread(strcat('../data/', test_imagenames{i}));
    vw = strcat(test_imagenames{i}(1:end-4), '.mat');
    load(strcat('../data/', vw));
    h1 = getImageFeatures(wordMap, K);
    for s=1:TrS
        h2 = trainFeatures(s, :);
        RDc(s) = getImageDistance(h1, h2, 'chi2');
        RDe(s) = getImageDistance(h1, h2, 'euclidean');     
    end
    [~, minRDc] = min(RDc);
    [~, minRDe] = min(RDe);
    label_c = trainLabels(minRDc);
    label_e = trainLabels(minRDe);
    
    realL = test_labels(i);
    
    %increment number correct
    if (label_c == realL)
        Rcorrect_c = Rcorrect_c + 1;
    end
    
    if (label_e == realL)
        Rcorrect_e = Rcorrect_e + 1;
    end
    
    %filling in confusion matrix   
    RCMc(realL, label_c) = RCMc(realL, label_c) + 1;
    RCMe(realL, label_e) = RCMc(realL, label_e) + 1;
end

fprintf('The accuracy of harris and chi-squared is %4.2f \n', Hcorrect_c/ImS);
fprintf('The confusion matrix for harris and chi-squared is: \n');
disp(HCMc);

fprintf('\n The accuracy of harris and euclidean is %4.2f \n', Hcorrect_e/ImS);
fprintf('The confusion matrix for harris and euclidean is: \n');
disp(HCMe);

fprintf('\n The accuracy of random and chi-squared is %4.2f \n', Rcorrect_c/ImS);
fprintf('The confusion matrix for random and chi-squared is: \n');
disp(RCMc);


fprintf('\n The accuracy of random and euclidean is %4.2f \n', Rcorrect_e/ImS);
fprintf('The confusion matrix for random and euclidean is: \n');
disp(RCMe);

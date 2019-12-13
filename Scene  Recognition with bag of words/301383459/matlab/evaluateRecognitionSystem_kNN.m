fprintf('Running kNN \n');
load('traintest.mat');
load('visionRandom.mat');
K = 100;

ImS = length(test_imagenames);
TrS = length(train_imagenames);

classes = zeros(ImS, 40);

correct_perK = zeros(40, 1);

RDc = zeros(TrS, 1);

confusion = zeros(8, 8);

%classify each image
for k = 1:40
    fprintf('Running kNN random, chisquared for k = %d\n', k);
    for i=1:ImS
        votes = zeros(8,1);
        
        I = imread(strcat('../data/', test_imagenames{i}));
        vw = strcat(test_imagenames{i}(1:end-4), '.mat');
        load(strcat('../data/', vw));
        h1 = getImageFeatures(wordMap, K);
        for s=1:TrS
            h2 = trainFeatures(s, :);
            RDc(s) = getImageDistance(h1, h2, 'chi2');   
        end
        
        %find k nearest neighbors
        [~, sortedI] = sort(RDc);
        knearest = sortedI(1:k);
        
        %loop through each neighbor and get its vote
        for n=1:k
            label = trainLabels(knearest(n));
            votes(label) = votes(label) + 1;
        end
        
        %find label with maximum votes
        [~, class] = max(votes);
        
        classes(i, k) = class;
        
        realL = test_labels(i);
        if realL == class
            correct_perK(k) = correct_perK(k) + 1;
        end
        
    end
end

[~, bestk] = max(correct_perK);
accuracies = correct_perK ./ ImS;

%fill in confusion matrix for bestK
for i=1:ImS 
    real = test_labels(i);   
    j = classes(i, bestk);
    confusion(real, j) = confusion(real, j) + 1;
end

fprintf('The confusion matrix for the best k is: \n');
disp(confusion);
plot(accuracies);
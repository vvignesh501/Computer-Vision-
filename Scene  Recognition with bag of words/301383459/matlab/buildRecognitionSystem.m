fprintf('Running 2.3 \n');
load('../data/traintest.mat', 'train_imagenames', 'train_labels');
trainLabels = train_labels';
T = length(train_imagenames);
K = 100;
trainFeatures = zeros(T, K);

fprintf('Loading Harris \n');
load('dictionaryHarris.mat', 'filterBank', 'dictionary');
for i=1:T
    I = imread(strcat('../data/', train_imagenames{i}));
    wordMap = getVisualWords(I, filterBank, dictionary');
    hist = getImageFeatures(wordMap, K);
    trainFeatures(i, :) = hist;
end
save('visionHarris.mat', 'dictionary', 'filterBank', 'trainFeatures','trainLabels');

fprintf('Loading Random \n');
load('dictionaryRandom.mat', 'filterBank', 'dictionary');
for i=1:T
    I = imread(strcat('../data/', train_imagenames{i}));
    wordMap = getVisualWords(I, filterBank, dictionary');
    hist = getImageFeatures(wordMap, K);
    trainFeatures(i, :) = hist;
end
save('visionRandom.mat', 'dictionary', 'filterBank', 'trainFeatures', 'trainLabels'); 
                          
fprintf('Finished \n');
                       
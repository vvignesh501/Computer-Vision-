fprintf('Running \n');
load('../data/traintest.mat', 'train_imagenames', 'train_labels');
trainLabels = train_labels';
K = 100;

fprintf('Loading Harris \n');
load('dictionaryHarris.mat', 'filterBank', 'dictionary');
%%Sample Image 1
I = imread('../data/airport/sun_afcsnnammgvyzrgm.jpg');
wordMap = getVisualWords(I, filterBank, dictionary');

%%Extract Filter Responses Output
response=extractFilterResponses(I,filterBank);
subplot(3, 1, 1);
imshow(response(:, :, 1), []);
title('L Image', 'FontSize', 20);
subplot(3, 1, 2);
imshow(response(:, :, 2), []);
title('A Image', 'FontSize', 20);
subplot(3, 1, 3);
imshow(response(:, :, 3), []);
title('B Image', 'FontSize', 20);

%% Get Visual Words Output for Harris
RGB2Harris = label2rgb(wordMap); 
figure;
imshow(RGB2Harris)

%%Sample Image 2
I = imread('../data/auditorium/sun_afcwsapxpihedtku.jpg');
wordMap = getVisualWords(I, filterBank, dictionary');
RGB2Harris = label2rgb(wordMap); 
figure;
imshow(RGB2Harris)

%%Sample Image 3
I = imread('../data/airport/sun_afrtpbxwfoubfcma.jpg');
wordMap = getVisualWords(I, filterBank, dictionary');
RGB2Harris = label2rgb(wordMap); 
figure;
imshow(RGB2Harris)


%%%% Random Loading
fprintf('Loading Random \n');
load('dictionaryRandom.mat', 'filterBank', 'dictionary');

%%Sample Image 1
I = imread('../data/airport/sun_afcsnnammgvyzrgm.jpg');
wordMap = getVisualWords(I, filterBank, dictionary');

%% Get Visual Words Output for Random
RGB2Random = label2rgb(wordMap); 
figure;
imshow(RGB2Random)

%%Sample Image 2
I = imread('../data/auditorium/sun_afcwsapxpihedtku.jpg');
wordMap = getVisualWords(I, filterBank, dictionary');
RGB2Random = label2rgb(wordMap); 
figure;
imshow(RGB2Random)


%%Sample Image 3
I = imread('../data/airport/sun_afrtpbxwfoubfcma.jpg');
wordMap = getVisualWords(I, filterBank, dictionary');
RGB2Random = label2rgb(wordMap); 
figure;
imshow(RGB2Random)

fprintf('Finished \n');
                       
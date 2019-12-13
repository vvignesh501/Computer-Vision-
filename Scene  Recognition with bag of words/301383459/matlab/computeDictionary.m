fprintf('Loading data \n');
load('../data/traintest.mat');
filterBank = createFilterBank();

dictionary = getDictionary(train_imagenames, 50, 100, 'random');
save('dictionaryRandom.mat', 'filterBank', 'dictionary');
fprintf('Finished loading random \n');

dictionary = getDictionary(train_imagenames, 50, 100, 'harris');
save('dictionaryHarris.mat', 'filterBank', 'dictionary');

fprintf('Finished');
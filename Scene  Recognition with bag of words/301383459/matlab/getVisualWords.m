function [wordMap] = getVisualWords(I, filterBank, dictionary)

dictionary = dictionary';
res = extractFilterResponses(I, filterBank);

rows = size(I, 1);
cols = size(I, 2);
n = size(filterBank, 1);

res = reshape(res, rows*cols, 3*n);


D = pdist2(res, dictionary, 'euclidean');

[~,closestK] = min(D,[],2);

wordMap = reshape(closestK, rows, cols);

end
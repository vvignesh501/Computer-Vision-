function [dictionary] = getDictionary(imgPaths, alpha, K, method)

T = size(imgPaths, 2);
bank = createFilterBank();
num_f = size(bank, 1);
pixelRes = zeros(alpha*T, 3*num_f);

for i=1:T
    I = imread(strcat('../data/', imgPaths{i}));
    %I = imread('../data/airport/sun_aerprlffjscovbbc.jpg');
    filter_res = extractFilterResponses(I, bank);
    if (strcmp(method, 'random'))
        points = getRandomPoints(I, alpha);
    else
        points = getHarrisPoints( I,alpha,K );
    end
    for point_row = 1:alpha
        dict_row = point_row + (i-1)*alpha;
        feats = filter_res(points(point_row, 1), points(point_row, 2), :);
        feat_v = reshape(feats, 1, 3*num_f);
        pixelRes(dict_row, :) = feat_v;
        save('pixelRes.mat', 'pixelRes');
    end
end
[~, dictionary] = kmeans(pixelRes, K, 'EmptyAction', 'drop');
end
function [points] = getRandomPoints(I, alpha)

points = zeros(alpha, 2);
rows = size(I, 1);
cols = size(I, 2);

ran_i = randi(rows*cols,alpha,1);

[points(:, 1), points(:, 2)] = ind2sub(size(I), ran_i);
end
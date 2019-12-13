function [points] = getHarrisPoints(I, alpha, k)

points = zeros(alpha, 2);

if numel(size(I)) > 2
    I = rgb2gray(I);
end
I = double(I);

sh = fspecial('sobel');
sv = transpose(sh);

Ix = imfilter(I, sh);
Iy = imfilter(I, sv);

Ixx = Ix .* Ix;
Iyy = Iy .* Iy;
Ixy = Ix .* Iy;

window = ones(5, 5);

as = imfilter(Ixx, window);
bs = imfilter(Ixy, window);
cs = imfilter(Ixy, window);
ds = imfilter(Iyy, window);

trace_sq = k*((as + ds).^2);
det = (as .* ds) - (cs .* bs);
Rs = det - trace_sq;

[~,sortIndex] = sort(Rs(:),'descend');  
                                                
maxIndex = sortIndex(1:alpha);

[points(:, 1), points(:, 2)] = ind2sub(size(I), maxIndex);

end
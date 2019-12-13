
function [ x, y, scores, Ix, Iy ] = getrandHarrisPoints( image )

G = rgb2gray(image);

% convert to double
G2 = im2double(G);

% create X and Y Sobel filters
horizontal_filter = [1 0 -1; 2 0 -2; 1 0 -1];
vertical_filter = [1 2 1; 0 0 0 ; -1 -2 -1];

% using imfilter to get our gradient in each direction
filtered_x = imfilter(G2, horizontal_filter);
filtered_y = imfilter(G2, vertical_filter);

% store the values in our output variables, for clarity
Ix = filtered_x;
Iy = filtered_y;


f = fspecial('gaussian');
Ix2 = imfilter(Ix.^2, f);
Iy2 = imfilter(Iy.^2, f);
Ixy = imfilter(Ix.*Iy, f);

% set empirical constant between 0.04-0.06
k = 0.04;

num_rows = size(image,1);
num_cols = size(image,2);

% create a matrix to hold the Harris values
H = zeros(num_rows, num_cols);

% % get our matrix M for each pixel
for y = 6:size(image,1)-6
    for x = 6:size(image,2)-6

        Ix2_matrix = Ix2(y-2:y+2,x-2:x+2);
        Ix2_mean = mean(Ix2_matrix(:));
        
        % Iy2 mean
        Iy2_matrix = Iy2(y-2:y+2,x-2:x+2);
        Iy2_mean = mean(Iy2_matrix(:));
        
        % Ixy mean
        Ixy_matrix = Ixy(y-2:y+2,x-2:x+2);
        Ixy_mean = mean(Ixy_matrix(:));
        
        Matrix = [Ix2_mean, Ixy_mean; 
                  Ixy_mean, Iy2_mean];
        R1 = det(Matrix) - (k * trace(Matrix)^2);
                
        H(y,x) = R1;
       
    end
end

% set threshold of 'cornerness' to 5 times average R score
avg_r = mean(mean(H))
threshold = abs(5 * avg_r)

[row, col] = find(H > threshold);

scores = [];
%get all the values
for index = 1:size(row,1)
    r = row(index);
    c = col(index);
    scores = cat(2, scores,H(r,c));
end

y = row;
x = col;

end

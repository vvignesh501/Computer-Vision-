function [lines, bw] = findLetters(im)

if size(im,3)==3
    im_gray = rgb2gray(im);
im_double = im2double(im_gray);
im_binary = imbinarize(im_double);
im_comp = imcomplement(im_binary);
SE_d = strel('sphere', 8);
SE_e = strel('disk', 8);
im_dil = imdilate(im_comp, SE_d);
im_erod = imerode(im_dil, SE_e);
im_bw = bwareaopen(im_erod, 50);

% Finding objects
CC = bwconncomp(im_bw);
im_size = CC.ImageSize;

% Finding bounding boxes
bounding_box = zeros(CC.NumObjects,4);
for i=1:CC.NumObjects
    [r,c] = ind2sub(im_size, CC.PixelIdxList{i});
    x_min = min(c);
    x_max = max(c);
    y_min = min(r);
    y_max = max(r);
    h = y_max - y_min;
    w = x_max - x_min;
    bounding_box(i,:) = [x_min, y_min, x_max, y_max]; 
end

% Sorting based on y_min
bounding_box_sorted = sortrows(bounding_box,2);

% Assigning line number to boxes
num_lines=1;
box_lines = zeros(CC.NumObjects,5);
box_lines(1,:) = [bounding_box_sorted(1,:),1];
for i=1:size(bounding_box_sorted)-1
    if bounding_box_sorted(i,4)<bounding_box_sorted(i+1,2)
        num_lines = num_lines+1;
    end
    box_lines(i+1,:) = [bounding_box_sorted(i+1,:),num_lines];
end

% Storing sorted lines and boxes in the cell
lines={};
box_lines = sortrows(box_lines,[5,1]);
for i=1:num_lines
    ind = box_lines(:,5) == i;
    lines{i} = box_lines(ind,1:4);
end

bw = im_binary;

end

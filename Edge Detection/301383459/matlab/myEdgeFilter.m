function [img1] = myEdgeFilter(img0, sigma)
%Your implemention

hsize = 2 * ceil(3 * sigma) + 1;
h = fspecial('gaussian',hsize,sigma);

filtered_img = myImageFilter(img0, h);
%Filter for horizontal and vertical direction

x = [-1, 0, 1; -2, 0, 2; -1, 0, 1];
y = [1, 2, 1; 0, 0, 0; -1, -2, -1];

%Convolution by image by horizontal and vertical filter
imgx = myImageFilter(filtered_img, x);
imgy = myImageFilter(filtered_img, y);

%Calculate directions
directions = atan2 (imgy, imgx);
directions = directions*180/pi;

size1=size(filtered_img,1);
size2=size(filtered_img,2);

for i=1:size1
    for j=1:size2
        if (directions(i,j)<0) 
            directions(i,j)=360+directions(i,j);
        end
    end
end

dir_adj=zeros(size1, size2);

%Adjusting directions to nearest 0, 45, 90, or 135 degree
for i = 1  : size1
    for j = 1 : size2
        if ((directions(i, j) >= 0 ) && (directions(i, j) < 22.5) || (directions(i, j) >= 157.5) && (directions(i, j) < 202.5) || (directions(i, j) >= 337.5) && (directions(i, j) <= 360))
            dir_adj(i, j) = 0;
        elseif ((directions(i, j) >= 22.5) && (directions(i, j) < 67.5) || (directions(i, j) >= 202.5) && (directions(i, j) < 247.5))
            dir_adj(i, j) = 45;
        elseif ((directions(i, j) >= 67.5 && directions(i, j) < 112.5) || (directions(i, j) >= 247.5 && directions(i, j) < 292.5))
            dir_adj(i, j) = 90;
        elseif ((directions(i, j) >= 112.5 && directions(i, j) < 157.5) || (directions(i, j) >= 292.5 && directions(i, j) < 337.5))
            dir_adj(i, j) = 135;
        end
    end
end

figure, imagesc(dir_adj);

%Calculate magnitude
magnitude = (imgx.^2) + (imgy.^2);
img_mag = sqrt(magnitude);

img1 = zeros (size1, size2);

%Non-Maximum Supression
for i=2:size1-1
    for j=2:size2-1
        if (dir_adj(i,j)==0)
            img1(i,j) = (img_mag(i,j) == max([img_mag(i,j), img_mag(i,j+1), img_mag(i,j-1)]));
        elseif (dir_adj(i,j)==45)
            img1(i,j) = (img_mag(i,j) == max([img_mag(i,j), img_mag(i+1,j-1), img_mag(i-1,j+1)]));
        elseif (dir_adj(i,j)==90)
            img1(i,j) = (img_mag(i,j) == max([img_mag(i,j), img_mag(i+1,j), img_mag(i-1,j)]));
        elseif (dir_adj(i,j)==135)
            img1(i,j) = (img_mag(i,j) == max([img_mag(i,j), img_mag(i+1,j+1), img_mag(i-1,j-1)]));
        end
    end
end

img1 = img1.*img_mag;
figure, imshow(img1);


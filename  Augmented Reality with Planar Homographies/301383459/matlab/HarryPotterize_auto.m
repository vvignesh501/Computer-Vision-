%Q2.2.4
clear all;
close all;

cv_img = imread('../data/cv_cover.jpg');
desk_img = imread('../data/cv_desk.png');
hp_img = imread('../data/hp_cover.jpg');
hp_desk = imread('../data/hp_desk.png');

%% Extract features and match
[locs1, locs2] = matchPics(cv_img, desk_img);

locs1=locs1.Location;
locs2=locs2.Location;

%% Compute homography using RANSAC
[bestH2to1, inliers_img1, inliers_img2, dist, index] = computeH_ransac(locs1, locs2);

figure;
desk_img=hp_desk;
showMatchedFeatures(cv_img,desk_img,inliers_img1,inliers_img2,'montage');

%% Scale harry potter image to template size
% Why is this is important?
scaled_hp_img = imresize(hp_img, [size(cv_img,1) size(cv_img,2)]);

%% Display warped image.
bestH2to1=double(bestH2to1);
imshow(warpH(cv_img, inv(bestH2to1), size(desk_img)));

%% Display composite image
imshow(compositeH(inv(bestH2to1), scaled_hp_img, desk_img));


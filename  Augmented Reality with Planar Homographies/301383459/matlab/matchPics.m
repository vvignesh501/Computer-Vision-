function [ locs1, locs2] = matchPics( I1, I2 )
%MATCHPICS Extract features, obtain their descriptors, and match them!

%% Convert images to grayscale, if necessary

if size(I1, 3) > 1
  new_Img1 = rgb2gray(I1);
  else
    new_Img1 = I1;

end


if size(I2, 3) > 1
  new_Img2 = rgb2gray(I2);
  else
    new_Img2 = I2;

end


%% Detect features in both images

det1 = detectSURFFeatures(new_Img1);
%strongest1 = detect1.selectStrongest(500);
%det1=strongest1.Location;
det2 = detectSURFFeatures(new_Img2);
%strongest2 = detect2.selectStrongest(500);
%det2=strongest2.Location;

%% Obtain descriptors for the computed feature locations

[features1,valid_points1]=extractFeatures(new_Img1,det1);
[features2,valid_points2]=extractFeatures(new_Img2,det2);


%% Match features using the descriptors

indexPairs = matchFeatures(features1,features2, 'MatchThreshold', 50, 'MaxRatio', 0.8);

locs1 = valid_points1(indexPairs(:,1),:);
locs2 = valid_points2(indexPairs(:,2),:);

figure; 
showMatchedFeatures(new_Img1,new_Img2,locs1,locs2,'montage');

warning('off', 'Images:initSize:adjustingMag');
end


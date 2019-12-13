% Your solution to Q2.1.5 goes here!

%% Read the image and convert to grayscale, if necessary

%% Compute the features and descriptors
cv_img = imread('../data/cv_cover.jpg');
counts=zeros(36,2);
angel=1;
for i = 0:36
    %% Rotate image
    angel=angel*10;
    img2=imrotate(cv_img,10);
    [locs1,locs2]=matchPics(cv_img,img2);
    if mod(angel,90)==0
        f=figure;
        fname=sprintf('rotational%i',angel);
        showMatchFeatures(cv_img,img2,locs1,locs2,'montage');
    end   
        counts(i+1,1)=i*10;
        counts(i+1,2)=size(locs1,1);
end

%% Match features using the descriptors

%% Display histogram

f=figure;
bar(counts(:,1),counts(:,2))
fname=sprintf('histogram');
title('SURF');
xlabel('Angle');
ylabel('Matches');
saveas(f,sprintf('../results/%s.png',fname));
close(f)

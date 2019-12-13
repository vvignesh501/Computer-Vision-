
clear;
clc;

im1 = imread('../data/im1.png');
im2 = imread('../data/im2.png');
load('../data/someCorresp.mat');



M = max(size(im1));

% compute fundamental matrix from corresponding points
F = eightpoint(pts1, pts2, M);


load('../data/templeCoords.mat');
load('../data/intrinsics.mat');

x1=pts1(:,1);
y1=pts1(:,2);

%displayEpipolarF(im1,im2,F)

% compute essential matrix from fundamental matrix and camera intrinsics
E = essentialMatrix(F, K1, K2);
%epipolarMatchGUI(im1,im2,F);
[n,~] = size(x1);
x2 = zeros(n, 1);
y2 = zeros(n, 1);


for i = 1:n
    [x, y] = epipolarCorrespondence(im1, im2, F, x1(i), y1(i));
    x2(i) = x;
    y2(i) = y;
end

P2 = camera2(E);

P1 = [eye(3), zeros(3,1)];

point1 = [x1,y1];
point2 = [x2,y2];

for i = 1:4
    P_ = triangulate(K1*P1, point1, K2*P2(:,:,i), point2);
    if all(P_(:,3) > 0)
        P = P_;
        M2 = P2(:,:,i);
    end
end

disp(P1)
disp(M2)

R1=P1(:,1:3);
R2=M2(:,1:3);

t1=P1(:,4);
t2=M2(:,4);

save('../data/extrinsics.mat', 'R1', 'R2', 't1', 't2');

P1 = K1*P1;
M2 = K2*M2;

scatter3(P(:,1),P(:,2),P(:,3), 'filled');


function [M2s] = camera2(E)

[U,S,V] = svd(E);

m = (S(1,1)+S(2,2))/2;

E = U*[m,0,0;
       0,m,0;
       0,0,0]*V';

[U,S,V] = svd(E);

W = [0,-1,0;
    1,0,0;
    0,0,1];

Z = [0 1 0; -1 0 0; 0 0 0];

R2= U * W' * V';
% Possible rotation matrices

% Force rotations to be proper, i. e. det(R) = 1
if det(U * W * V') < 0
    W = -W;
end

if det(R2) < 0
    R2 = -R2;
end

M2s = zeros(3,4,4);
M2s(:,:,1) = [U*W*V',U(:,3)./max(abs(U(:,3)))];
M2s(:,:,2) = [U*W*V',-U(:,3)./max(abs(U(:,3)))];
M2s(:,:,3) = [R2,U(:,3)./max(abs(U(:,3)))];
M2s(:,:,4) = [R2,-U(:,3)./max(abs(U(:,3)))];


end

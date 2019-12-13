function [pts3d] = triangulate(P1, pts1, P2, pts2 )

[n1,~] = size(pts1);
[n2,~] = size(pts2);

pts1 = [pts1'; ones(1,n1)];
pts2 = [pts2'; ones(1,n2)];

P = zeros(4,n1);

for i=1:n1
	x = [0 pts1(3,i) -pts1(2,i); -pts1(3,i) 0 pts1(1,i); pts1(2,i) -pts1(1,i) 0];
	y = [0 pts2(3,i) -pts2(2,i); -pts2(3,i) 0 pts2(1,i); pts2(2,i) -pts2(1,i) 0];

	Q = [x*P1; y*P2];
	[~,~,V] = svd(Q);
	z = V(:,end);
	P(:,i) = z/z(4);
end

pts3d = P(1:3,:)';

p1_hat = P1*P;
p2_hat = P2*P;

error = (pts1-p1_hat).^2 + (pts2-p2_hat).^2;

disp(error)

end

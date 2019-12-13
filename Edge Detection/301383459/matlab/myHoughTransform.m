function [H, rhoScale, thetaScale] = myHoughTransform(Im, threshold, rhoRes, thetaRes)

Im=edge(Im,'sobel',threshold);
[y,x]=find(Im > threshold);
[sy,sx]=size(Im);
tot_pixel = length(x);
max_rho = round(sqrt(sx^2 + sy^2));

H = zeros(2*max_rho,180);

for i = 1:tot_pixel
i2 = 1;
for thetaScale = 0:thetaRes*1/2:pi-thetaRes*1/2
rhoScale = round(x(i).*cos(thetaScale) + y(i).*sin(thetaScale));
H(rhoScale+max_rho,i2) = H(rhoScale+max_rho,i2) + 1;
i2 = i2 + 1;
end
end

thetaScale = rad2deg(0:thetaRes*1/2:pi-thetaRes*1/2);
rhoScale = -max_rho:rhoRes:max_rho-1;
warning('off', 'Images:initSize:adjustingMag');
imshow(uint8(H),[],'xdata',thetaScale,'ydata',rhoScale);
xlabel('\theta'),ylabel('\rho')
axis on, axis normal;
title('Hough Matrix');


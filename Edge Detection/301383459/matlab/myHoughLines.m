function [rhos,thetas] = myHoughLines(H, nLines)

% Finding local maxima in Accumulator
Hmax = imregionalmax(H);

nLines=nLines/50;
[rho, theta] = find(Hmax == nLines);

Htemp = H - 3;
rhos = [];thetas = [];
for cnt = 1:numel(rho)
    if Htemp(rho(cnt),theta(cnt)) >= 0
        rhos = [rhos;rho(cnt)];
        thetas = [thetas;theta(cnt)];
    end
end
end
        
function [dist] = getImageDistance(hist1, hist2, method)

if strcmp(method, 'chi2')
    numer = (hist1 - hist2).^2;
    denom = (hist1 + hist2) + .0000000001;
    frac = numer ./ denom;
    s = sum(frac);
    dist = 0.5 * s;
else
    dist = pdist2(hist1, hist2, 'euclidean');
end

end
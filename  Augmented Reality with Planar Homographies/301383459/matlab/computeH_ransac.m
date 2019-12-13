function [bestH2to1, maxinliers_img1, maxinliers_img2, max_pt_line_dist, index] = computeH_ransac(image1_points, image2_points)
%
h = size(image1_points, 1);
N = 2000;
inlier_dist_thresh = 3;
inlier_maxcount = -99;
random_pts = 10;
bestH2to1 = [];

for iterations=1:N
    
    %find 10 random pixels in both image1 and image2
    n1 = numel(image1_points(:,1));
    idx = randperm(n1,random_pts);
    src = image1_points(idx,:);
    dest = image2_points(idx,:);
    
    F = computeH_norm([src, dest]);
    inliers_img1 = [];
    inliers_img2 = [];
    L = (F * [image1_points ones(h,1)]')'; 
    L = L ./ repmat(sqrt(L(:,1).^2 + L(:,2).^2), 1, 3); 
    pt_line_dist = abs(sum(L .* [image2_points ones(h,1)],2));
    index = find(pt_line_dist < inlier_dist_thresh);
    inliers_img1 = [inliers_img1;image1_points(index, :)];
    inliers_img2 = [inliers_img2;image2_points(index,:)];

    if(~isempty(inliers_img1)) 
        if(numel(inliers_img1(:,1)) > inlier_maxcount)
            bestH2to1 = F;
            max_pt_line_dist = pt_line_dist;
            maxinliers_img1 = inliers_img1;
            maxinliers_img2 = inliers_img2;
            inlier_maxcount = numel(inliers_img1(:,1));
        end
    end
end

end
function [ composite_img ] = compositeH( H2to1, template, img )
%COMPOSITE Create a composite image after warping the template image on top
%of the image using the homography

% Note that the homography we compute is from the image to the template;
% x_template = H2to1*x_photo
% For warping the template to the image, we need to invert it.
H_template_to_img = inv(H2to1);

%% Create mask of same size as template
mask=(template> 0.5);
%% Warp mask by appropriate homography
mask_warp=warpH(mask,H2to1,size(img));
%% Warp template by appropriate homography
template_warp=warpH(template,H2to1,size(img));
%% Use mask to combine the warped template and the image

composite_img=template_warp+img;
imshow(composite_img)

end
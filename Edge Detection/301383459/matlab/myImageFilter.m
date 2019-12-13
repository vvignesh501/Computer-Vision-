function [img1] = myImageFilter(img0, h) 
[r,c] = size(img0);

[m,n] = size(h);
h = rot90(h, 2);
img1=zeros(size(img0));
img0= padarray(img0,[floor(m*0.5) floor(n*0.5)],'replicate','pre');
img0= padarray(img0,[floor(m*0.5) floor(n*0.5)],'replicate','post');


for i= 1:r
    for j=1:c
        tempA= img0(i:i+m-1,j:j+n-1).*h;
        img1(i,j)=sum(tempA(:));        
    end
end
end
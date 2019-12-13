function dispM = get_disparity(im1, im2, maxDisp, windowSize)
% GET_DISPARITY creates a disparity map from a pair of rectified images im1 and
%   im2, given the maximum disparity MAXDISP and the window size WINDOWSIZE.

w = (windowSize-1)/2 ;
md = maxDisp ;
dispM=zeros(size(im1,1),size(im1,2)) ;
for i=md+w+1:size(im1,1)-(md+w+1)
    for j=w+1:size(im1,2)-(w+1)
        for d=0:maxDisp
            val(d+1)=norm(double(im1(i-w:i+w,j-w:j+w) - im2(i-w-d:i+w-d,j-w:j+w))) ;
        end
        dispM(i,j)=find(val==min(val),1) ;
    end
    
end
end
function [filterResponses] = extractFilterResponses(I, filterBank)

I = im2double(I);

if numel(size(I)) < 3
    I(:, :, 1) = I;
    I(:, :, 2) = I(:, :, 1);
    I(:, :, 3) = I(:, :, 1);
end

R = I(:, :, 1);
G = I(:, :, 2);
B = I(:, :, 3);
[L, a, b] = RGB2Lab(R, G, B);
I(:, :, 1) = L;
I(:, :, 2) = a;
I(:, :, 3) = b;


num_fs = size(filterBank, 1);

filterResponses = zeros(size(I,1), size(I,2), 3*num_fs); 
        
for f = 1:num_fs
  cur_f = filterBank{f};
  for c = 1:3
    page = (f-1)*3 + c;
    filterResponses(:, :, page) = imfilter(I(:, :, c), cur_f);
  end
end

end
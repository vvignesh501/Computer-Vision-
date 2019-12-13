function ec()
im={};    
im{1} = imread('../images/image1.jpg');
im{2} = imread('../images/image2.jpg');
im{3} = imread('../images/image3.png');
im{4} = imread('../images/image4.jpg');


input_im = im{1};
imshow(input_im);
hold on;
[lines, ~] = findLetters(input_im);
for j=1:size(lines,2)
    curr_line = lines{j};
    for k=1:size(curr_line,1)
        x_min = curr_line(k,1);
        y_min = curr_line(k,2);
        w = curr_line(k,3)-curr_line(k,1);
        h = curr_line(k,4)-curr_line(k,2);
        rectangle('Position',[x_min y_min w h]);
    end
end


end
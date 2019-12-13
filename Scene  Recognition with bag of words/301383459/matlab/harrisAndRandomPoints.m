%%%Harris Points generation
image1 = imread('../data/campus/sun_abslhphpiejdjmpz.jpg');
image2=  imread('../data/airport/sun_aerinlrdodkqnypz.jpg');
image3=  imread('../data/bedroom/sun_aesmnitwkuadtpxy.jpg');


%Image 1
[ x, y, scores, Ix, Iy ] = getrandHarrisPoints( image1 );
figure; imshow(image1) 
hold on
for i = 1:size(scores,2)
    plot(x(i), y(i), 'ro', 'MarkerSize', scores(i) * 0.1);
 end
saveas(gcf,'../results/harris_corners_sample1.png');
hold off

harris1= imread('../results/harris_corners_sample1.png');
figure;imshow(harris1);


%Image 2
[ x, y, scores, Ix, Iy ] = getrandHarrisPoints( image2 );
figure; imshow(image2) 
hold on
for i = 1:size(scores,2)
    plot(x(i), y(i), 'ro', 'MarkerSize', scores(i) * 0.1)    
end
saveas(gcf,'../results/harris_corners_sample2.png');
hold off

harris2= imread('../results/harris_corners_sample2.png');
figure;imshow(harris2);

%Image 3
[ x, y, scores, Ix, Iy ] = getrandHarrisPoints( image3 );
figure; imshow(image3) 
hold on
for i = 1:size(scores,2)
    plot(x(i), y(i), 'ro', 'MarkerSize', scores(i) * 0.1);
end
saveas(gcf,'../results/harris_corners_sample3.png');
hold off

harris3= imread('../results/harris_corners_sample3.png');
figure;imshow(harris3);

%%%%Random Points generation

image4 = imread('../data/campus/sun_abslhphpiejdjmpz.jpg');
image5=  imread('../data/airport/sun_aerinlrdodkqnypz.jpg');
image6=  imread('../data/bedroom/sun_aesmnitwkuadtpxy.jpg');

[ points ] = getRandomPoints( image4, 500 );
figure; imshow(image4) 

hold on
plot(points(:,2), points(:,1), 'ro', 'MarkerSize', 1)
saveas(gcf,'../results/random_corners_sample1.png');
hold off


[ points ] = getRandomPoints( image5, 500 );
figure; imshow(image5) 

hold on
plot(points(:,2), points(:,1), 'ro', 'MarkerSize', 1)
saveas(gcf,'../results/random_corners_sample2.png');
hold off


[ points ] = getRandomPoints( image6, 500 );
figure; imshow(image6) 

hold on
plot(points(:,2), points(:,1), 'ro', 'MarkerSize', 1)
saveas(gcf,'../results/random_corners_sample3.png');
hold off



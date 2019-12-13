image1 = imread('../data/campus/sun_abslhphpiejdjmpz.jpg');

image2=  imread('../data/airport/sun_aerprlffjscovbbc.jpg');
image3=  imread('../data/bedroom/sun_aesmnitwkuadtpxy.jpg');

[ points ] = getRandomPoints( image1, 500 );
figure; imshow(image1) 
hold on

plot(points(:,1), points(:,2), 'ro', 'MarkerSize', 1)
saveas(gcf,'random_corners_sample1.png');
hold off


[ points ] = getRandomPoints( image2, 500 );
figure; imshow(image2) 
hold on

plot(points(:,1), points(:,2), 'ro', 'MarkerSize', 1)
saveas(gcf,'random_corners_sample2.png');
hold off


[ points ] = getRandomPoints( image3, 500 );
figure; imshow(image3) 
hold on

plot(points(:,1), points(:,2), 'ro', 'MarkerSize', 1)
saveas(gcf,'random_corners_sample3.png');
hold off
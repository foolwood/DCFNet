im = imread('peppers.png');
im = imresize(im,[256,256]);
x = double(fhog(single(im) / 255, 4, 9));
x(:,:,end) = [];
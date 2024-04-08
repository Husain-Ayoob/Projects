input_image = imread('peppers.tif');
load('watermark.mat', 'w1', 'w2', 'w3');

input_image = double(input_image);

dct_image = dct2(input_image);

w1_location = [128, 128];
w2_location = [128, 256];
w3_location = [256, 128];

alpha = 1.5;
dct_image(w1_location(1):w1_location(1)+127, w1_location(2):w1_location(2)+127) = dct_image(w1_location(1):w1_location(1)+127, w1_location(2):w1_location(2)+127) + w1 * alpha;

dct_image(w2_location(1):w2_location(1)+127, w2_location(2):w2_location(2)+127) = dct_image(w2_location(1):w2_location(1)+127, w2_location(2):w2_location(2)+127) + w2 * alpha;

dct_image(w3_location(1):w3_location(1)+127, w3_location(2):w3_location(2)+127) = dct_image(w3_location(1):w3_location(1)+127, w3_location(2):w3_location(2)+127) + w3 * alpha;

watermarked_image = idct2(dct_image);

imwrite(uint8(watermarked_image), 'WatermarkedPeppersDCT.tif', 'tif');

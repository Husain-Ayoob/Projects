watermarked_image = imread('WatermarkedPeppersDCT.tif');
load('watermark.mat', 'w1', 'w2', 'w3'); %Ask Martin Keep watermark for testing?
dct_watermarked_image = dct2(watermarked_image);

w1_location = [128, 128];
w2_location = [128, 256];
w3_location = [256, 128];

detection_region1 = dct_watermarked_image(w1_location(1):w1_location(1)+127, w1_location(2):w1_location(2)+127);
detection_region2 = dct_watermarked_image(w2_location(1):w2_location(1)+127, w2_location(2):w2_location(2)+127);
detection_region3 = dct_watermarked_image(w3_location(1):w3_location(1)+127, w3_location(2):w3_location(2)+127);

phi1 = calculate_phi(detection_region1, w1);
sigma1 = std(detection_region1(:));
threshold1 = sigma1 * 10; 

phi2 = calculate_phi(detection_region2, w2);
sigma2 = std(detection_region2(:));
threshold2 = sigma2 * 10; 

phi3 = calculate_phi(detection_region3, w3);
sigma3 = std(detection_region3(:));
threshold3 = sigma3 * 10; 

is_w1_present = (phi1 > threshold1); 
is_w2_present = (phi2 > threshold2); 
is_w3_present = (phi3 > threshold3); 

fprintf('Presence of w1: %d\n', is_w1_present);
fprintf('Presence of w2: %d\n', is_w2_present);
fprintf('Presence of w3: %d\n', is_w3_present);


function phi = calculate_phi(dct_watermarked_image, watermark)
    embed_col = dct_watermarked_image(:);
    watermark_col = watermark(:);

    phi = (sum(watermark_col .* sign(embed_col))).^2 / sum(watermark_col.^2);

end

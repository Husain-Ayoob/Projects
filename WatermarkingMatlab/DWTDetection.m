[watermarked_image, map] = imread('WatermarkedPeppersDWT.tif');

load('watermark.mat', 'w1', 'w2', 'w3');

[A1, H1, V1, D1] = dwt2(watermarked_image, 'haar');

[A2, H2, V2, D2] = dwt2(A1, 'haar');



phi1 = calculate_phi(H2, w1);
sigma1 = std(H2(:));
threshold1 = sigma1 * 10; 

phi2 = calculate_phi(V2, w2);
sigma2 = std(V2(:));
threshold2 = sigma2 * 10; 

phi3 = calculate_phi(D2, w3);
sigma3 = std(D2(:));
threshold3 = sigma3 * 10; 

watermark_detected = [phi1 > threshold1, phi2 > threshold2, phi3 > threshold3];

disp(['Watermark w1 detected in H2: ', num2str(watermark_detected(1))]);
disp(['Watermark w2 detected in D2: ', num2str(watermark_detected(2))]);
disp(['Watermark w3 detected in V2: ', num2str(watermark_detected(3))]);


function phi = calculate_phi(embed_watermark, watermark)
    embed_col = embed_watermark(:);
    watermark_col = watermark(:);

    phi = (sum(watermark_col .* sign(embed_col))).^2 / sum(watermark_col.^2);

end




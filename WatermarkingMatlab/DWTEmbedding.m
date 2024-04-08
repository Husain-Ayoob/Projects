original_image = imread('peppers.tif');

original_image_double = double(original_image);

[A1, H1, V1, D1] = dwt2(original_image_double, 'haar');

[A2, H2, V2, D2] = dwt2(A1, 'haar');

load('watermark.mat', 'w1', 'w2', 'w3');

alpha = 1.5;  % Watermark strength
H2_embedded = embed_watermark(H2, w1, alpha);
V2_embedded = embed_watermark(V2, w2, alpha);
D2_embedded = embed_watermark(D2, w3, alpha);
A1Water  = idwt2(A2, H2_embedded, V2_embedded, D2_embedded, 'haar');
watermarked_image = idwt2(A1Water, H1, V1, D1, 'haar');


watermarked_image1 = uint8(watermarked_image);

imwrite(watermarked_image1, 'WatermarkedPeppersDWT.tif');

function embedded_sub_band = embed_watermark(sub_band, watermark, alpha)
    assert(isequal(size(sub_band), size(watermark)), 'Size of sub_band and watermark must be the same.');
    
    embedded_sub_band = sub_band + alpha * watermark;
end

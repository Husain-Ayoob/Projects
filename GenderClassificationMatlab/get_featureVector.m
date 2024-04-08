function vector=get_featureVector(image)
% this function returns a column vector of 35 features extracted in the seventh level 
% of wavelet decomposition (Haar wavelet).
% First convert the colour input image into a greayscale image
image= rgb2gray(image);

[A1,H1,V1,D1] = dwt2(image,'haar'); %level 1
[A2,H2,V2,D2] = dwt2(A1,'haar');  %level 2
[A3,H3,V3,D3] = dwt2(A2,'haar'); 
[A4,H4,V4,D4] = dwt2(A3,'haar'); 
[A5,H5,V5,D5] = dwt2(A4,'haar'); 
[A6,H6,V6,D6] = dwt2(A5,'haar'); 
[A7,H7,V7,D7] = dwt2(A6,'haar'); % level 7 
vector=A7(:); % the approximation subband is used as features
end
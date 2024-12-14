% Step 1: Load the host image
host_image = imread('jonah-brown-lN1-wyjEVHQ-unsplash.jpg');
host_image = rgb2gray(host_image); % Convert to grayscale
host_image = im2uint8(imresize(host_image, [512, 512])); % Resize for simplicity

% Step 2: Load the watermark image
watermark = imread('Untitled.png');
watermark = rgb2gray(watermark); % Convert to grayscale
watermark = im2bw(imresize(watermark, [512, 512])); % Resize and binarize

% Step 3: Embed the watermark in the host image
watermarked_image = host_image;
watermarked_image = bitset(watermarked_image, 1, watermark); % Set LSB to the watermark bit

% Step 4: Save and display the watermarked image
imwrite(watermarked_image, 'watermarked_image.jpg');
figure;
subplot(1, 3, 1), imshow(host_image), title('Host Image');
subplot(1, 3, 2), imshow(watermark), title('Watermark');
subplot(1, 3, 3), imshow(watermarked_image), title('Watermarked Image');

% Step 5: Extract the watermark from the watermarked image
extracted_watermark = bitget(watermarked_image, 1); % Extract LSB
figure;
imshow(extracted_watermark), title('Extracted Watermark');


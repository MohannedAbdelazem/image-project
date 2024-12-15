# Image-Project
I used this code initially for text detection but it did not give good accuracy:
```octave
pkg load image  % Load image processing package

% Step 1: Load and preprocess the image
img = imread('images.jpeg'); % Replace with your image path
gray_img = rgb2gray(img);      % Convert to grayscale

% Step 2: Apply edge detection
edges = edge(gray_img, 'Canny'); % Use Canny edge detection

% Step 3: Morphological operations
se = strel('rectangle', [5, 5]); % Define structuring element
dilated_img = imdilate(edges, se); % Dilate to connect text-like regions

% Step 4: Find connected components
[labeled_img, num] = bwlabel(dilated_img); % Label connected components
stats = regionprops(labeled_img, 'BoundingBox'); % Get bounding boxes

% Step 5: Display results
imshow(img); hold on;
for k = 1 : num
    rectangle('Position', stats(k).BoundingBox, 'EdgeColor', 'r', 'LineWidth', 2);
end
hold off;

# Note
I changed the idea into a generic one for simplicity (specifically noise removal)

% Noise Removal for Colored Images using Mean and Median Filters

% Read the input image (colored)
input_image = imread('mean_noisy.jpeg');
% --------------------------
% Function: Mean Filter
% --------------------------
function output_image = noise_removal_mean(input_image)
    % Convert image to double for processing
    input_image = double(input_image);
    [rows, cols, channels] = size(input_image);

    % Initialize output image
    output_image = zeros(rows, cols, channels);

    % Define kernel size (3x3)
    kernel_size = 3;
    half_kernel = floor(kernel_size / 2);

    % Process each channel
    for c = 1:channels
        for i = 1+half_kernel : rows-half_kernel
            for j = 1+half_kernel : cols-half_kernel
                % Extract the neighborhood
                neighborhood = input_image(i-half_kernel:i+half_kernel, j-half_kernel:j+half_kernel, c);

                % Compute the mean
                mean_value = mean(neighborhood(:));

                % Assign mean value to the output image
                output_image(i, j, c) = mean_value;
            end
        end
    end

    % Convert back to uint8
    output_image = uint8(output_image);
endfunction
% --------------------------
% Function: Median Filter
% --------------------------
function output_image = noise_removal_median(input_image)
    % Convert image to double for processing
    input_image = double(input_image);
    [rows, cols, channels] = size(input_image);

    % Initialize output image
    output_image = zeros(rows, cols, channels);

    % Define kernel size (3x3)
    kernel_size = 3;
    half_kernel = floor(kernel_size / 2);

    % Process each channel
    for c = 1:channels
        for i = 1+half_kernel : rows-half_kernel
            for j = 1+half_kernel : cols-half_kernel
                % Extract the neighborhood
                neighborhood = input_image(i-half_kernel:i+half_kernel, j-half_kernel:j+half_kernel, c);

                % Compute the median
                median_value = median(neighborhood(:));

                % Assign median value to the output image
                output_image(i, j, c) = median_value;
            end
        end
    end

    % Convert back to uint8
    output_image = uint8(output_image);
endfunction
% Apply Mean Filter Noise Removal
output_mean = noise_removal_mean(input_image);

% Apply Median Filter Noise Removal
output_median = noise_removal_median(input_image);

% Display the results
figure;
subplot(1, 3, 1); imshow(input_image); title('Original Noisy Image');
subplot(1, 3, 2); imshow(output_mean); title('Denoised (Mean Filter)');
subplot(1, 3, 3); imshow(output_median); title('Denoised (Median Filter)');


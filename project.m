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
% --------------------------
% Function: Gaussian Filter
% --------------------------
function output_image = gaussian_filter(input_image)
    % Convert the input image to double
    input_image = double(input_image);

    % Define Gaussian kernel (3x3)
    gaussian_kernel = [1, 2, 1; 2, 4, 2; 1, 2, 1] / 16;

    % Add padding to handle boundaries
    padded_image = padarray(input_image, [1, 1], 'symmetric');

    % Get the size of the image
    [rows, cols] = size(input_image);

    % Initialize output image
    output_image = zeros(rows, cols);

    % Apply Gaussian filter
    for i = 1:rows
        for j = 1:cols
            % Extract the 3x3 neighborhood
            neighborhood = padded_image(i:i+2, j:j+2);

            % Apply Gaussian kernel
            output_image(i, j) = sum(sum(neighborhood .* gaussian_kernel));
        end
    end

    % Convert back to uint8 for display
    output_image = uint8(output_image);
endfunction
% --------------------------
% Function: Bilateral Filter
% --------------------------
function output_image = bilateral_filter(input_image, sigma_spatial, sigma_intensity)
    % Convert the input image to double
    input_image = double(input_image);

    % Define kernel size
    kernel_size = 3;
    half_kernel = floor(kernel_size / 2);

    % Add padding to handle boundaries
    padded_image = padarray(input_image, [half_kernel, half_kernel], 'symmetric');

    % Get the size of the image
    [rows, cols] = size(input_image);

    % Initialize output image
    output_image = zeros(rows, cols);

    % Precompute Gaussian spatial weights
    [X, Y] = meshgrid(-half_kernel:half_kernel, -half_kernel:half_kernel);
    spatial_weights = exp(-(X.^2 + Y.^2) / (2 * sigma_spatial^2));

    % Apply Bilateral filter
    for i = 1:rows
        for j = 1:cols
            % Extract the neighborhood
            neighborhood = padded_image(i:i+2*half_kernel, j:j+2*half_kernel);

            % Compute Gaussian intensity weights
            intensity_weights = exp(-(neighborhood - padded_image(i+half_kernel, j+half_kernel)).^2 / (2 * sigma_intensity^2));

            % Combine weights
            combined_weights = spatial_weights .* intensity_weights;

            % Normalize weights
            combined_weights = combined_weights / sum(combined_weights(:));

            % Apply weights to neighborhood
            output_image(i, j) = sum(sum(neighborhood .* combined_weights));
        end
    end

    % Convert back to uint8 for display
    output_image = uint8(output_image);
endfunction
% --------------------------
% Function: Wiener Filter
% --------------------------
function output_image = wiener_filter(input_image, kernel_size)
    % Convert the input image to double
    input_image = double(input_image);

    % Define kernel size
    half_kernel = floor(kernel_size / 2);

    % Add padding to handle boundaries
    padded_image = padarray(input_image, [half_kernel, half_kernel], 'symmetric');

    % Get the size of the image
    [rows, cols] = size(input_image);

    % Initialize output image
    output_image = zeros(rows, cols);

    % Apply Wiener filter
    for i = 1:rows
        for j = 1:cols
            % Extract the neighborhood
            neighborhood = padded_image(i:i+2*half_kernel, j:j+2*half_kernel);

            % Compute local mean and variance
            local_mean = mean(neighborhood(:));
            local_variance = var(neighborhood(:));

            % Compute noise variance (assume it's global for simplicity)
            noise_variance = 0.01; % You can adjust this value based on noise level

            % Wiener filter formula
            output_value = local_mean + max(0, local_variance - noise_variance) / max(local_variance, noise_variance) * (input_image(i, j) - local_mean);

            % Assign the result to the output image
            output_image(i, j) = output_value;
        end
    end

    % Convert back to uint8 for display
    output_image = uint8(output_image);
endfunction
% Apply Mean Filter Noise Removal
output_mean = noise_removal_mean(input_image);

% Apply Median Filter Noise Removal
output_median = noise_removal_median(input_image);
% Apply Gaussian Filter
output_gaussian = gaussian_filter(input_image);
% Apply Bilateral Filter (requires sigma values)
output_bilateral = bilateral_filter(input_image, 1.5, 25);
%Apply Wiener Filter (with a 3x3 kernel)
output_wiener = wiener_filter(input_image, 3);
% Display the results
figure;
subplot(2, 3, 1); imshow(input_image); title('Original Noisy Image');
subplot(2, 3, 2); imshow(output_mean); title('Denoised (Mean Filter)');
subplot(2, 3, 3); imshow(output_median); title('Denoised (Median Filter)');
subplot(2, 3, 4); imshow(output_gaussian); title('Denoised (gaussian Filter)');
subplot(2, 3, 5); imshow(output_bilateral); title('Denoised (bilateral Filter)');
subplot(2, 3, 6); imshow(output_wiener); title('Denoised (wiener Filter)');



% Noise Removal for Colored Images using Various Filters

% Read the input image (colored)
input_image = imread('gaussian_noisy.jpeg');


% ----------------------
% Function Definitions
% ----------------------

function output_image = noise_removal_mean(input_image)
    input_image = double(input_image);
    [rows, cols, channels] = size(input_image);
    kernel_size = 3;
    half_kernel = floor(kernel_size / 2);
    output_image = zeros(size(input_image));

    for c = 1:channels
        for i = 1+half_kernel : rows-half_kernel
            for j = 1+half_kernel : cols-half_kernel
                neighborhood = input_image(i-half_kernel:i+half_kernel, j-half_kernel:j+half_kernel, c);
                output_image(i, j, c) = mean(neighborhood(:));
            end
        end
    end
    output_image = uint8(output_image);
end

function output_image = noise_removal_median(input_image)
    input_image = double(input_image);
    [rows, cols, channels] = size(input_image);
    kernel_size = 3;
    half_kernel = floor(kernel_size / 2);
    output_image = zeros(size(input_image));

    for c = 1:channels
        for i = 1+half_kernel : rows-half_kernel
            for j = 1+half_kernel : cols-half_kernel
                neighborhood = input_image(i-half_kernel:i+half_kernel, j-half_kernel:j+half_kernel, c);
                output_image(i, j, c) = median(neighborhood(:));
            end
        end
    end
    output_image = uint8(output_image);
end

function output_image = gaussian_filter(input_image)
    % Convert input image to double for precise calculations
    input_image = double(input_image);
    [rows, cols, channels] = size(input_image);

    % Define a 7x7 Gaussian kernel
    gaussian_kernel = [
        0, 0, 1, 2, 1, 0, 0;
        0, 3, 13, 22, 13, 3, 0;
        1, 13, 59, 97, 59, 13, 1;
        2, 22, 97, 159, 97, 22, 2;
        1, 13, 59, 97, 59, 13, 1;
        0, 3, 13, 22, 13, 3, 0;
        0, 0, 1, 2, 1, 0, 0;
    ] / 1003;

    % Pad the input image symmetrically (2 pixels for 7x7 kernel)
    padded_image = padarray(input_image, [3, 3], 'symmetric');

    % Initialize the output image
    output_image = zeros(size(input_image));

    % Loop through each channel, row, and column of the image
    for c = 1:channels
        for i = 1:rows
            for j = 1:cols
                % Extract the 5x5 neighborhood
                neighborhood = padded_image(i:i+6, j:j+6, c);

                % Apply the Gaussian kernel
                output_image(i, j, c) = sum(sum(neighborhood .* gaussian_kernel));
            end
        end
    end

    % Convert the output image back to uint8 format
    output_image = uint8(output_image);
end


function output_image = bilateral_filter(input_image, sigma_spatial, sigma_intensity)
    input_image = double(input_image);
    [rows, cols, channels] = size(input_image);

    % Dynamically adjust kernel size based on sigma_spatial
    kernel_size = ceil(2 * sigma_spatial * 3);
    half_kernel = floor(kernel_size / 2);

    % Pad image symmetrically
    padded_image = padarray(input_image, [half_kernel, half_kernel], 'symmetric');

    % Prepare spatial weights
    [X, Y] = meshgrid(-half_kernel:half_kernel, -half_kernel:half_kernel);
    spatial_weights = exp(-(X.^2 + Y.^2) / (2 * sigma_spatial^2));

    % Initialize output image
    output_image = zeros(size(input_image));

    for c = 1:channels
        for i = 1:rows
            for j = 1:cols
                % Extract the neighborhood
                neighborhood = padded_image(i:i+2*half_kernel, j:j+2*half_kernel, c);

                % Compute intensity weights
                intensity_weights = exp(-(neighborhood - padded_image(i + half_kernel, j + half_kernel, c)).^2 / (2 * sigma_intensity^2));

                % Combine spatial and intensity weights
                combined_weights = spatial_weights .* intensity_weights;
                combined_weights = combined_weights / sum(combined_weights(:));  % Normalize

                % Apply the bilateral filter
                output_image(i, j, c) = sum(sum(neighborhood .* combined_weights));
            end
        end
    end

    % Ensure pixel values are in the correct range
    output_image = uint8(max(0, min(255, output_image)));
end


function output_image = wiener_filter(input_image, kernel_size, noise_variance)
    input_image = double(input_image);
    [rows, cols, channels] = size(input_image);
    half_kernel = floor(kernel_size / 2);
    padded_image = padarray(input_image, [half_kernel, half_kernel], 'symmetric');
    output_image = zeros(size(input_image));

    for c = 1:channels
        for i = 1:rows
            for j = 1:cols
                neighborhood = padded_image(i:i+2*half_kernel, j:j+2*half_kernel, c);
                local_mean = mean(neighborhood(:));
                local_variance = var(neighborhood(:));

                % Wiener filter equation
                output_image(i, j, c) = local_mean + ...
                    max(0, local_variance - noise_variance) / max(local_variance, noise_variance) * (input_image(i, j, c) - local_mean);
            end
        end
    end

    % Clamping output to [0, 255] and converting to uint8
    output_image = uint8(max(0, min(255, output_image)));
end


function output_image = adaptive_median_filter(input_image, S_max)
    input_image = double(input_image);
    [rows, cols, channels] = size(input_image);
    output_image = zeros(size(input_image));

    for c = 1:channels
        for i = 1:rows
            for j = 1:cols
                S = 3;
                while S <= S_max
                    row_start = max(1, i-floor(S/2));
                    row_end = min(rows, i+floor(S/2));
                    col_start = max(1, j-floor(S/2));
                    col_end = min(cols, j+floor(S/2));
                    window = input_image(row_start:row_end, col_start:col_end, c);

                    Zmin = min(window(:));
                    Zmax = max(window(:));
                    Zmed = median(window(:));

                    if Zmed > Zmin && Zmed < Zmax
                        output_image(i, j, c) = Zmed;
                        break;
                    elseif S > S_max
                        output_image(i, j, c) = input_image(i, j, c);
                    else
                        S = S + 2;
                    end
                end
            end
        end
    end
    output_image = uint8(output_image);
end

function output_image = lee_filter(input_image, window_size)
    input_image = double(input_image);
    [rows, cols, channels] = size(input_image);
    half_window = floor(window_size / 2);
    output_image = zeros(size(input_image));

    for c = 1:channels
        for i = 1+half_window : rows-half_window
            for j = 1+half_window : cols-half_window
                window = input_image(i-half_window:i+half_window, j-half_window:j+half_window, c);
                local_mean = mean(window(:));
                local_variance = var(window(:));
                global_variance = var(input_image(:));
                coefficient = local_variance / (local_variance + global_variance);
                output_image(i, j, c) = local_mean + coefficient * (input_image(i, j, c) - local_mean);
            end
        end
    end
    output_image = uint8(output_image);
end

function output_image = alpha_trimmed_mean_filter(input_image, alpha)
    input_image = double(input_image);
    [rows, cols, channels] = size(input_image);
    kernel_size = 3;
    half_kernel = floor(kernel_size / 2);
    output_image = zeros(size(input_image));

    for c = 1:channels
        for i = 1+half_kernel : rows-half_kernel
            for j = 1+half_kernel : cols-half_kernel
                neighborhood = input_image(i-half_kernel:i+half_kernel, j-half_kernel:j+half_kernel, c);
                sorted_neighborhood = sort(neighborhood(:));
                num_trim = round(alpha * numel(neighborhood));
                trimmed_neighborhood = sorted_neighborhood(num_trim+1:end-num_trim);

                if ~isempty(trimmed_neighborhood)
                    output_image(i, j, c) = mean(trimmed_neighborhood);
                else
                    output_image(i, j, c) = input_image(i, j, c);
                end
            end
        end
    end
    output_image = uint8(output_image);
end


function output_image = non_local_means_filter(input_image, search_window, patch_size, h)
    input_image = double(input_image);
    [rows, cols, channels] = size(input_image);
    half_patch = floor(patch_size / 2);
    half_search = floor(search_window / 2);

    % Initialize the output image
    output_image = zeros(size(input_image));

    % Iterate through each pixel in the image
    for c = 1:channels
        for i = 1+half_patch : rows-half_patch
            for j = 1+half_patch : cols-half_patch
                % Extract the patch centered at (i, j)
                patch = input_image(i-half_patch:i+half_patch, j-half_patch:j+half_patch, c);

                % Initialize the weights and weighted sum for the pixel
                weights = zeros(rows, cols);
                weighted_sum = 0;
                weight_sum = 0;

                % Search window: loop through all other patches in the search window
                for m = max(1, i-half_search) : min(rows, i+half_search)
                    for n = max(1, j-half_search) : min(cols, j+half_search)
                        % Ensure the current patch stays within bounds
                        if m-half_patch >= 1 && m+half_patch <= rows && n-half_patch >= 1 && n+half_patch <= cols
                            % Extract the current patch from the search window
                            current_patch = input_image(m-half_patch:m+half_patch, n-half_patch:n+half_patch, c);

                            % Compute the Euclidean distance between patches
                            distance = norm(patch - current_patch, 'fro');

                            % Compute the similarity (Gaussian kernel of the distance)
                            similarity = exp(-distance^2 / (h^2));

                            % Update the weights and the weighted sum
                            weighted_sum = weighted_sum + similarity * input_image(m, n, c);
                            weight_sum = weight_sum + similarity;
                        end
                    end
                end

                % Compute the denoised value for the current pixel
                if weight_sum > 0
                    output_image(i, j, c) = weighted_sum / weight_sum;
                else
                    output_image(i, j, c) = input_image(i, j, c);  % If no valid patches, keep original pixel value
                end
            end
        end
    end

    % Convert the output image back to uint8
    output_image = uint8(output_image);
end


function output_image = kalman_filter(input_image, process_noise, measurement_noise, search_window)
    input_image = double(input_image);
    [rows, cols, channels] = size(input_image);

    % Initialize variables
    estimate = input_image;  % Initial estimate
    error_covariance = ones(rows, cols, channels);  % Initial error covariance
    half_window = floor(search_window / 2);

    % Convergence tolerance
    tolerance = 1e-3;  
    max_iters = 10;

    for iter = 1:max_iters
        previous_estimate = estimate;  % Store previous estimate for convergence check

        for c = 1:channels
            for i = 1:rows
                for j = 1:cols
                    % Define the search window bounds
                    row_min = max(1, i - half_window);
                    row_max = min(rows, i + half_window);
                    col_min = max(1, j - half_window);
                    col_max = min(cols, j + half_window);

                    % Aggregate measurements within the search window
                    neighboring_values = input_image(row_min:row_max, col_min:col_max, c);
                    measured_value = mean(neighboring_values(:));  % Average over the window

                    % Prediction
                    predicted_estimate = estimate(i, j, c);
                    predicted_covariance = error_covariance(i, j, c) + process_noise;

                    % Kalman Gain
                    kalman_gain = predicted_covariance / (predicted_covariance + measurement_noise);

                    % Update estimate and error covariance
                    estimate(i, j, c) = predicted_estimate + kalman_gain * (measured_value - predicted_estimate);
                    error_covariance(i, j, c) = (1 - kalman_gain) * predicted_covariance;
                end
            end
        end

        % Check for convergence
        if max(abs(estimate(:) - previous_estimate(:))) < tolerance
            break;
        end
    end

    % Clip and convert to uint8
    output_image = uint8(min(max(estimate, 0), 255));
end

function output_image = tvd_filter(input_image, lambda, num_iterations)
    input_image = double(input_image);
    [rows, cols, channels] = size(input_image);

    % Initialize variables
    output_image = input_image;
    gradient_x = zeros(rows, cols, channels);  % x-gradient of the image
    gradient_y = zeros(rows, cols, channels);  % y-gradient of the image
    norm_gradient = zeros(rows, cols, channels); % To store the norm of gradients

    % Perform iterations to reduce the total variation
    for iter = 1:num_iterations
        % Compute the gradients for each channel separately
        for c = 1:channels
            % Calculate the x-gradient
            gradient_x(2:end, :, c) = output_image(2:end, :, c) - output_image(1:end-1, :, c);  % x-gradient for channel c

            % Calculate the y-gradient
            gradient_y(:, 2:end, c) = output_image(:, 2:end, c) - output_image(:, 1:end-1, c);  % y-gradient for channel c
        end

        % Compute the norm of the gradient for each channel
        norm_gradient = sqrt(gradient_x.^2 + gradient_y.^2);

        % Update the image by reducing the total variation
        for c = 1:channels
            for i = 1:rows
                for j = 1:cols
                    if norm_gradient(i,j,c) > 0
                        % Update the pixel value based on the gradient
                        output_image(i,j,c) = output_image(i,j,c) - lambda * (gradient_x(i,j,c) + gradient_y(i,j,c));
                    end
                end
            end
        end
    end

    % Convert the output image back to uint8
    output_image = uint8(min(max(output_image, 0), 255));
end

% Apply Filters
output_mean = noise_removal_mean(input_image);
output_median = noise_removal_median(input_image);
output_gaussian = gaussian_filter(input_image);
output_bilateral = bilateral_filter(input_image, 3, 25);
output_wiener = wiener_filter(input_image, 7, 0.1);
output_adaptive = adaptive_median_filter(input_image, 7);
output_lee = lee_filter(input_image, 5);
output_alpha_trimmed = alpha_trimmed_mean_filter(input_image, 0.2);
output_nlm = non_local_means_filter(input_image, 7, 5, 10);
output_kalman = kalman_filter(input_image, 0.1, 0.5, 7);
output_tvd = tvd_filter(input_image, 0.1, 50);

% Display the results
figure;
subplot(3, 4, 1); imshow(input_image); title('Original Noisy Image');
subplot(3, 4, 2); imshow(output_mean); title('Mean Filter');
subplot(3, 4, 3); imshow(output_median); title('Median Filter');
subplot(3, 4, 4); imshow(output_gaussian); title('Gaussian Filter');
subplot(3, 4, 5); imshow(output_bilateral); title('Bilateral Filter');
subplot(3, 4, 6); imshow(output_wiener); title('Wiener Filter');
subplot(3, 4, 7); imshow(output_adaptive); title('Adaptive Median Filter');
subplot(3, 4, 8); imshow(output_lee); title('Lee Filter');
subplot(3, 4, 9); imshow(output_alpha_trimmed); title('Alpha-Trimmed Filter');
subplot(3, 4, 10); imshow(output_nlm); title('NLM Filter');
subplot(3, 4, 11); imshow(output_tvd); title('TVD Filter');
subplot(3, 4, 12); imshow(output_kalman); title('Kalman Filter');
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

% --------------------------
% Function: Adaptive Median Filter
% --------------------------
function output_image = adaptive_median_filter(input_image, S_max)
  % Convert image to double for processing
  input_image = double(input_image);
  [rows, cols, channels] = size(input_image);

  % Initialize output image
  output_image = zeros(rows, cols, channels);

  % Process each channel
  for c = 1:channels
    for i = 1:rows
      for j = 1:cols
        S = 3; % Initial window size
        Zxy = input_image(i, j, c);
        Zmed = 0;

        while S <= S_max
          % Extract the neighborhood
          row_start = max(1, i - floor(S/2));
          row_end = min(rows, i + floor(S/2));
          col_start = max(1, j - floor(S/2));
          col_end = min(cols, j + floor(S/2));
          window = input_image(row_start:row_end, col_start:col_end, c);

          % Calculate Zmin, Zmax, and Zmed
          Zmin = min(window(:));
          Zmax = max(window(:));
          Zmed = median(window(:));

          % Level A
          if Zmed > Zmin && Zmed < Zmax
            output_image(i, j, c) = Zmed;
            break;
          end

          % Level B
          if Zmed == Zmin || Zmed == Zmax
            S = S + 2; % Increase window size
          end

          % Level C
          if S > S_max
            output_image(i, j, c) = Zmed;
          end
        end
      end
    end
  end

  % Convert back to uint8
  output_image = uint8(output_image);
endfunction

% --------------------------
% Function: Lee Filter
% --------------------------
function output_image = lee_filter(input_image, window_size)
  % Convert the input image to double
  input_image = double(input_image);

  % Get image dimensions
  [rows, cols, channels] = size(input_image);

  % Initialize output image
  output_image = zeros(size(input_image));

  % Half the window size
  half_window_size = floor(window_size / 2);

  % Iterate through each pixel
  for c = 1:channels
    for i = 1+half_window_size:rows-half_window_size
      for j = 1+half_window_size:cols-half_window_size

        % Extract the local window
        window = input_image(i-half_window_size:i+half_window_size, j-half_window_size:j+half_window_size, c);

        % Calculate local mean and variance
        local_mean = mean(window(:));
        local_variance = var(window(:));

        % Calculate global mean and variance (you can adjust these values)
        global_mean = 128;  % Example value, adjust based on your image
        global_variance = 100;  % Example value, adjust based on your image

        % Calculate the filter coefficient
        coefficient = local_variance / (local_variance + global_variance);

        % Apply the filter
        output_image(i, j, c) = local_mean + coefficient * (input_image(i, j, c) - local_mean);

      end
    end
  end

  % Clip pixel values to the valid range
  output_image = min(max(output_image, 0), 255);

  % Convert back to uint8 for display
  output_image = uint8(output_image);
endfunction

% --------------------------
% Function: Alpha-Trimmed Mean Filter
% --------------------------
function output_image = alpha_trimmed_mean_filter(input_image, alpha)
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

        % Sort the neighborhood
        sorted_neighborhood = sort(neighborhood(:));

        % Calculate the number of values to trim from each end
        num_trim = round(alpha * numel(neighborhood));

        % Remove the 'alpha' percentage of highest and lowest values
        trimmed_neighborhood = sorted_neighborhood(num_trim+1:end-num_trim);

        % Calculate the mean of the remaining values
        mean_value = mean(trimmed_neighborhood);

        % Assign the mean value to the output image
        output_image(i, j, c) = mean_value;
      end
    end
  end

  % Convert back to uint8
  output_image = uint8(output_image);
endfunction

% Apply Mean Filter
output_mean = noise_removal_mean(input_image);

% Apply Median Filter
output_median = noise_removal_median(input_image);

% Apply Gaussian Filter
output_gaussian = gaussian_filter(input_image);

% Apply Bilateral Filter (requires sigma values)
output_bilateral = bilateral_filter(input_image, 1.5, 25);

%Apply Wiener Filter (with a 3x3 kernel)
output_wiener = wiener_filter(input_image, 3);

% Apply Adaptive Median Filter
output_adaptive = adaptive_median_filter(input_image, 7);

% Apply Lee Filter
output_lee = lee_filter(input_image, 5);

% Apply Alpha-Trimmed Mean Filter
output_alpha_trimmed = alpha_trimmed_mean_filter(input_image, 0.2);

% Display the results
figure;
subplot(4, 3, 1); imshow(input_image); title('Original Noisy Image');
subplot(4, 3, 2); imshow(output_mean); title('Denoised (Mean Filter)');
subplot(4, 3, 3); imshow(output_median); title('Denoised (Median Filter)');
subplot(4, 3, 4); imshow(output_gaussian); title('Denoised (Gaussian Filter)');
subplot(4, 3, 5); imshow(output_bilateral); title('Denoised (Bilateral Filter)');
subplot(4, 3, 6); imshow(output_wiener); title('Denoised (Wiener Filter)');
subplot(4, 3, 7); imshow(output_adaptive); title('Denoised (Adaptive Median Filter)');
subplot(4, 3, 8); imshow(output_lee); title('Denoised (Lee Filter)');
subplot(4, 3, 9); imshow(output_alpha_trimmed); title('Denoised (Alpha-Trimmed Mean Filter)');


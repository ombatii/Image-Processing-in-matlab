
% Specify the target directory
targetDirectory = 'C:\Users\OMBATI\Desktop\matlab\project - 3';

% Change the current working directory
cd(targetDirectory);

% Display the current working directory to verify the change
currentDirectory = pwd;
disp(['Current working directory: ' currentDirectory]);


% Question 1 (15 Marks)
% Original histogram data
r_k = 0:7;
n_k = [1028, 3544, 5023, 3201, 1867, 734, 604, 383];

% Compute cumulative distribution function (CDF)
cdf = cumsum(n_k) / sum(n_k);

% Compute equalized intensity values
value = round((2^3 - 1) * cdf);

% Apply the point operation to equalize the image
equalized_image = zeros(size(r_k));
for i = 1:length(r_k)
    equalized_image(i) = value(r_k(i) + 1);
end

% Plot the original histogram
figure;
subplot(2, 1, 1);
bar(r_k, n_k, 'g'); 
title('Original Histogram');
xlabel('Intensity Levels');
ylabel('Frequency');

% Plot the equalized histogram
subplot(2, 1, 2);
bar(r_k, equalized_image, 'k'); 
title('Equalized Histogram');
xlabel('Intensity Levels');
ylabel('Equalized Intensity Values');

% Display the original and equalized histograms
legend('Original Histogram', 'Equalized Histogram', 'Location', 'Northwest'); % Move the legend to the top left
% Question 2 (15 Marks)
% Given histogram
rk_given = 0:7;
nk_given = [0, 0, 4096, 4096, 4096, 4096, 0, 0];

% Target histogram
rk_target = 0:7;
nk_target = [0, 1365, 2731, 4096, 4096, 2731, 1365, 0];

% Compute cumulative distribution functions (CDFs) for given and target histograms
cdf_given = cumsum(nk_given) / sum(nk_given);
cdf_target = cumsum(nk_target) / sum(nk_target);

% Find the mapping function (point operation)
s = zeros(size(rk_given));
for i = 1:length(rk_given)
    [~, index] = min(abs(cdf_given(i) - cdf_target));
    s(i) = rk_target(index);
end

% Apply the point operation to match histograms
matched_image = zeros(size(rk_given));
for i = 1:length(rk_given)
    matched_image(i) = s(rk_given(i) + 1);
end

% Plot the given, target, and matched histograms
figure;

subplot(3, 1, 1);
bar(rk_given, nk_given, 'FaceColor', 'black'); % Set color to black
title('Given Histogram');
xlabel('Intensity Levels');
ylabel('Frequency');

subplot(3, 1, 2);
bar(rk_target, nk_target, 'FaceColor', 'blue'); % Set color to blue
title('Target Histogram');
xlabel('Intensity Levels');
ylabel('Frequency');

subplot(3, 1, 3);
bar(rk_given, matched_image, 'FaceColor', 'red'); % Set color to red
title('Matched Histogram');
xlabel('Intensity Levels');
ylabel('Matched Intensity Values');

% Display the histograms with legend in the top-left corner
legend('Given Histogram', 'Target Histogram', 'Matched Histogram', 'Location', 'NorthWest');


% Question 4 (20 Marks)
% Load the original image
lena_gray_original_image = imread('lena_gray_512.tif');

% Display the original image
figure;
subplot(2, 4, 1);
imshow(lena_gray_original_image);
title('lena_gray_Original Image');

% Densities of salt & pepper noise
densities = [0.05, 0.1, 0.2];

% Filter sizes
filter_sizes = [3, 5];

% Process the images with salt & pepper noise and apply filters
for density_index = 1:length(densities)
    for filter_size_index = 1:length(filter_sizes)
        % Add salt & pepper noise
        noisy_image = imnoise(lena_gray_original_image, 'salt & pepper', densities(density_index));

        % Apply median filter
        median_filtered_image = medfilt2(noisy_image, [filter_sizes(filter_size_index), filter_sizes(filter_size_index)]);

        % Apply smoothing average filter
        smoothing_average_filtered_image = imfilter(noisy_image, ones(filter_sizes(filter_size_index))/(filter_sizes(filter_size_index)^2));

        % Display the noisy image
        subplot(2, 4, density_index + 1);
        imshow(noisy_image);
        title(['Noisy Image (Density: ' num2str(densities(density_index)) ')']);

        % Display the median filtered image
        subplot(2, 4, density_index + 4 + filter_size_index);
        imshow(median_filtered_image);
        title(['Median Filter (' num2str(filter_sizes(filter_size_index)) 'x' num2str(filter_sizes(filter_size_index)) ')']);

        % Display the smoothing average filtered image
        subplot(2, 4, density_index + 8 + filter_size_index);
        imshow(smoothing_average_filtered_image);
        title(['Smoothing Avg Filter (' num2str(filter_sizes(filter_size_index)) 'x' num2str(filter_sizes(filter_size_index)) ')']);
    end
end

% Adjust the figure layout
sgtitle('Image Processing with Salt & Pepper Noise and Filters');


% Question 3 (50 Marks)

% Specify the target directory
targetDirectory = 'C:\Users\OMBATI\Desktop\matlab\project - 3';

% Change the current working directory
cd(targetDirectory);

% Display the current working directory to verify the change
currentDirectory = pwd;
disp(['Current working directory: ' currentDirectory]);

% Process "livingroom.tif" with both equalize and histeq functions
livingroom_original = imread('livingroom.tif');

% Check if the image is grayscale or RGB
if size(livingroom_original, 3) == 3
    % Convert RGB image to grayscale
    livingroom_gray = rgb2gray(livingroom_original);
else
    % Image is already grayscale
    livingroom_gray = livingroom_original;
end

% Using custom equalize function
livingroom_equalized_custom = equalize(livingroom_gray);

% Using histeq function
livingroom_equalized_histeq = histeq(livingroom_gray);

% Display the images before and after equalization
figure;
subplot(3, 2, 1); imshow(livingroom_gray); title('Living Room Original');
subplot(3, 2, 2); imshow(livingroom_equalized_custom); title('Equalized (Custom)');
subplot(3, 2, 3); imhist(livingroom_gray); title('Original Histogram');
subplot(3, 2, 4); imhist(livingroom_equalized_custom); title('Equalized Histogram (Custom)');
subplot(3, 2, 5); imshow(livingroom_equalized_histeq); title('Equalized (histeq)');
subplot(3, 2, 6); imhist(livingroom_equalized_histeq); title('Equalized Histogram (histeq)');

% Process "woman_darkhair.tif" with both equalize and histeq functions
woman_original = imread('woman_darkhair.tif');

% Check if the image is grayscale or RGB
if size(woman_original, 3) == 3
    % Convert RGB image to grayscale
    woman_gray = rgb2gray(woman_original);
else
    % Image is already grayscale
    woman_gray = woman_original;
end

% Using custom equalize function
woman_equalized_custom = equalize(woman_gray);

% Using histeq function
woman_equalized_histeq = histeq(woman_gray);

% Display the images before and after equalization
figure;
subplot(3, 2, 1); imshow(woman_gray); title('Woman Original');
subplot(3, 2, 2); imshow(woman_equalized_custom); title('Equalized (Custom)');
subplot(3, 2, 3); imhist(woman_gray); title('Original Histogram');
subplot(3, 2, 4); imhist(woman_equalized_custom); title('Equalized Histogram (Custom)');
subplot(3, 2, 5); imshow(woman_equalized_histeq); title('Equalized (histeq)');
subplot(3, 2, 6); imhist(woman_equalized_histeq); title('Equalized Histogram (histeq)');

function im2 = equalize(im)
    % Check if the input image is uint8
    if ~isa(im, 'uint8')
        error('Input image must be uint8.');
    end
    
    % Get the histogram of the input image
    hist_im = imhist(im);

    % Compute the cumulative distribution function (CDF) of the histogram
    cdf = cumsum(hist_im) / sum(hist_im);

    % Perform histogram equalization
    T = uint8((2^8 - 1) * cdf);
    im2 = T(im + 1); % Apply the transformation

    % Note: The '+1' is used to map intensity values from the range [0, 255] to [1, 256]
end








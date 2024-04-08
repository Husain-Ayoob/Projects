% Initialize counters
tp_nn = 0; fp_nn = 0; fn_nn = 0; tn_nn = 0; % For Neural Network
tp_lp = 0; fp_lp = 0; fn_lp = 0; tn_lp = 0; % For Linear Perceptron
tp_svm = 0; fp_svm = 0; fn_svm = 0; tn_svm = 0; % For Linear SVM
tp_tree = 0; fp_tree = 0; fn_tree = 0; tn_tree = 0; % For Classification Tree

% Define the folder containing the test images
folder_path = 'test_gender\';
image_files = dir(fullfile(folder_path, '*.jpg')); % Adjust the file extension if needed

% Loop over each image in the folder
for i = 1:length(image_files)
    file_name = fullfile(folder_path, image_files(i).name);
    test_image = imread(file_name);
    feature_vector = get_featureVector(test_image); % Get the feature vector
    is_female = startsWith(image_files(i).name, 'female'); % Check if the image is of a female


    % Linear Perceptron Testing
    output_perceptron = nets(feature_vector);
    if output_perceptron 
        if is_female, tp_lp = tp_lp + 1; else, fp_lp = fp_lp + 1; end
    else
        if is_female, fn_lp = fn_lp + 1; else, tn_lp = tn_lp + 1; end
    end

    % Linear SVM Testing
    Y_testing_svm = predict(svm, feature_vector');
    if Y_testing_svm
        if is_female, tp_svm = tp_svm + 1; else, fp_svm = fp_svm + 1; end
    else
        if is_female, fn_svm = fn_svm + 1; else, tn_svm = tn_svm + 1; end
    end

    % Classification Tree Testing
    Y_testing_tree = predict(tree, feature_vector');
    if Y_testing_tree
        if is_female, tp_tree = tp_tree + 1; else, fp_tree = fp_tree + 1; end
    else
        if is_female, fn_tree = fn_tree + 1; else, tn_tree = tn_tree + 1; end
    end
end

% Calculate FPR, FNR and their average for each classifier
fpr_nn = fp_nn / (fp_nn + tn_nn);
fnr_nn = fn_nn / (fn_nn + tp_nn);
avg_nn = (fpr_nn + fnr_nn) / 2;

fpr_lp = fp_lp / (fp_lp + tn_lp);
fnr_lp = fn_lp / (fn_lp + tp_lp);
avg_lp = (fpr_lp + fnr_lp) / 2;

fpr_svm = fp_svm / (fp_svm + tn_svm);
fnr_svm = fn_svm / (fn_svm + tp_svm);
avg_svm = (fpr_svm + fnr_svm) / 2;

fpr_tree = fp_tree / (fp_tree + tn_tree);
fnr_tree = fn_tree / (fn_tree + tp_tree);
avg_tree = (fpr_tree + fnr_tree) / 2;

% Display the results
fprintf('Neural Network: FPR = %.2f, FNR = %.2f, Average = %.2f\n', fpr_nn, fnr_nn, avg_nn);
fprintf('Linear Perceptron: FPR = %.2f, FNR = %.2f, Average = %.2f\n', fpr_lp, fnr_lp, avg_lp);
fprintf('Linear SVM: FPR = %.2f, FNR = %.2f, Average = %.2f\n', fpr_svm, fnr_svm, avg_svm);
fprintf('Classification Tree: FPR = %.2f, FNR = %.2f, Average = %.2f\n', fpr_tree, fnr_tree, avg_tree);

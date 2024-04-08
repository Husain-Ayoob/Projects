% testing phase - Test Images are in folder test_gender - let's read the
% first one
clc;

file_name=['test_gender\female17.jpg'];

test_image=imread(file_name);
feature_vector=get_featureVector(test_image); % function get_featureVector returns a column vector.


% Linear Perceptron Testing
output_perceptron = nets(feature_vector);

if output_perceptron 
    disp('Linear Perceptron: female');
else
    disp('Linear Perceptron: male');
end

% Linear SVM Testing
Y_testing_svm = predict(svm, feature_vector');
if Y_testing_svm
    disp('Linear SVM: female');
else
    disp('Linear SVM: male');
end

% Classification Tree Testing
Y_testing_tree = predict(tree, feature_vector');
if Y_testing_tree
    disp('Classification Tree: female');
else
    disp('Classification Tree: male');
end

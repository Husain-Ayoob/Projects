% Load the data
data = spamhamdataset;



% Ensure that the table contains the necessary columns
if ~all(ismember({'CATEGORY', 'MESSAGE'}, data.Properties.VariableNames))
    error('Error: Input file must contain columns "CATEGORY" and "MESSAGE".');
end

%summary(data)

data.Properties.VariableNames = {'CATEGORY', 'MESSAGE', 'FILE_NAME'};
fprintf('Size of data: %d\n', size(data));

% Preprocess the data
documents = data.MESSAGE;
fprintf('Size of documents: %d\n', size(documents));
documents = lower(documents); % Convert all text to lowercase
fprintf('Size of documents after lowercase: %d\n', size(documents));
documents = erasePunctuation(documents); % Remove punctuation
fprintf('Size of documents after punctuation removal: %d\n', size(documents));
documents = tokenizedDocument(documents);
fprintf('Size of documents after tokenization: %d\n', size(documents));
documents = removeStopWords(documents); % Remove stop words
fprintf('Size of documents after stopword removal: %d\n', size(documents));
documents = normalizeWords(documents,'Style','stem'); % Stem the words
fprintf('Size of documents after stemming: %d\n', size(documents));
bag = bagOfWords(documents); % Convert the preprocessed documents into a bag of words
fprintf('Size of bag: %d\n', size(bag));

% Calculate the TF-IDF weights for the bag of words
tfidf_weights = tfidf(bag); % Rename variable to avoid conflicts
fprintf('Size of tfidf_weights: %d\n', size(tfidf_weights));

% Convert the bag of words and TF-IDF weights into a table
tbl = array2table(full(tfidf_weights)); % Remove variable names
var_names = strcat('word_', string(1:size(tbl,2))); % Create variable names
tbl.Properties.VariableNames = var_names; % Set variable names
msg_table = tbl; % rename the table to a more descriptive name
fprintf('Size of tbl: %d\n', size(tbl));

% Add the target variable to the table
tbl.Category = categorical(data.CATEGORY);
fprintf('Size of tbl after adding Category: %d\n', size(tbl));

% Split the data into training, validation, and testing sets
try
cvp = cvpartition(tbl.Category,'HoldOut',0.3);
cvp_train = cvpartition(cvp.TrainSize,'HoldOut',0.2); % 60% for training
idx_train = find(cvp_train.training);
idx_validate = find(cvp_train.test);
idx_test = find(cvp.test);
tbl_train = tbl(idx_train,:);
tbl_validate = tbl(idx_validate,:);
tbl_test = tbl(idx_test,:);
catch
error('Error: Could not split data into training, validation, and testing sets.');
end




fprintf('Size of cvp: %d\n', size(cvp));
tblTrain = tbl(cvp.training,:); % this is a table data structure
fprintf('Size of tblTrain: %d\n', size(tblTrain));
tblTest = tbl(cvp.test,:);


fprintf('Size of tblTest: %d\n', size(tblTest));
X_train = full(table2array(tblTrain(:, 1:end-1)));
Y_train = tblTrain.Category;

if size(X_train, 2) ~= size(tblTest(:,1:end-1), 2)
    error('Error: The number of columns in X_train does not match the number of columns in tblTest(:,1:end-1).');
end

% Check for missing values in tblTrain
if sum(ismissing(tblTrain)) > 0
    warning('Warning: Training data contains missing values. Removing rows with missing data.');
    tblTrain = rmmissing(tblTrain);
end

% Check for missing values in tblTest
if sum(ismissing(tblTest)) > 0
    warning('Warning: Testing data contains missing values. Removing rows with missing data.');
    tblTest = rmmissing(tblTest);
end

% Train the classifier

try
    Mdl = fitcsvm(X_train,Y_train); % fitcsvm , fitctree
catch
    error('Error: Could not train classifier.');
end


% Predict the labels of the testing set
X_test = full(table2array(tblTest(:, 1:end-1)));

predictedLabels_temp = predict(Mdl, X_test);
fprintf('Size of predictedLabels_temp: %d\n', numel(predictedLabels_temp));
predictedLabels = predictedLabels_temp;
fprintf('Size of predictedLabels: %d\n', numel(predictedLabels));
% Get the actual labels of the testing set
trueLabels = categorical(tblTest.Category);
% Evaluate the performance of the classifier
accuracy = sum(predictedLabels == tblTest.Category)/numel(tblTest.Category);
fprintf('Accuracy: %f\n', accuracy);
% Calculate the confusion matrix
confMat = confusionmat(trueLabels, predictedLabels);

% Calculate precision, recall, and F1 score
precision = confMat(2,2)/(confMat(2,2)+confMat(1,2));
fprintf('precision: %f\n', precision);
recall = confMat(2,2)/(confMat(2,2)+confMat(2,1));
fprintf('recall: %f\n', recall);
f1Score = 2*(precision*recall)/(precision+recall);
fprintf('f1Score: %f\n', f1Score);


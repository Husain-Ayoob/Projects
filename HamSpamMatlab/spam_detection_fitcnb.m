% Load the data
data = spamhamdataset;

% Ensure that the table contains the necessary columns
if ~all(ismember({'CATEGORY', 'MESSAGE'}, data.Properties.VariableNames))
    error('Error: Input file must contain columns "CATEGORY" and "MESSAGE".');
end

% Preprocess the data
documents = data.MESSAGE;
documents = lower(documents); % Convert all text to lowercase
documents = erasePunctuation(documents); % Remove punctuation
documents = tokenizedDocument(documents);
documents = removeStopWords(documents); % Remove stop words
documents = normalizeWords(documents,'Style','stem'); % Stem the words

% Convert the preprocessed documents into a bag of words
bag = bagOfWords(documents);

% Calculate the TF-IDF weights for the bag of words
tfidf_weights = tfidf(bag);

% Convert the bag of words and TF-IDF weights into a table
tbl = array2table(full(tfidf_weights));
var_names = strcat('word_', string(1:size(tbl,2))); % Create variable names
tbl.Properties.VariableNames = var_names; % Set variable names
tbl.Category = categorical(data.CATEGORY); % Add the target variable to the table


% Remove any variable with zero variance
X = data{:, 2:end-1};
X(:, var(X)==0) = [];
Y = data{:, end};

% Split data into training and testing sets
cv = cvpartition(size(X,1),'HoldOut',0.3);
idx = cv.test;
X_train = X(~idx,:);
Y_train = Y(~idx,:);
X_test = X(idx,:);
Y_test = Y(idx,:);

% Train Naive Bayes classifier
Mdl = fitcnb(X_train, Y_train);

% Predict on testing set
Y_pred = predict(Mdl, X_test);

% Evaluate performance
confMat = confusionmat(Y_test, Y_pred);
accuracy = sum(diag(confMat))/sum(confMat(:));
precision = confMat(2,2)/(confMat(2,2)+confMat(1,2));
recall = confMat(2,2)/(confMat(2,2)+confMat(2,1));
f1 = 2*precision*recall/(precision+recall);
fprintf('Accuracy: %.2f\n', accuracy)
fprintf('Precision: %.2f\n', precision)
fprintf('Recall: %.2f\n', recall)
fprintf('F1 score: %.2f\n', f1)

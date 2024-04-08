clear;
import ClassificationTree.*
folder_male=dir('Male\');
folder_female=dir('Female\');
Number_male_images=length(folder_male)-2; % number of male images
Number_female_images=length(folder_female)-2; % number of female images
input_training_set=[];

% Construct the training set 
% each training feature vector (either for Male or Female images) is extracted 
% by using the DWT. Please see function get_featureVector
% dimension of the classifier is 35 (35 features)
for i=1:Number_male_images
    Image=imread(['Male\' folder_male(i+2).name]);
    input_training_set=[input_training_set;get_featureVector(Image)'];   
    output_training_set{i,1}='male';
end
k=i;
for i=1:Number_female_images
    Image=imread(['Female\' folder_female(i+2).name]);
    input_training_set=[input_training_set;get_featureVector(Image)'];
    output_training_set{k+i,1}='female';
   
end
% because we have 70 training samples, the input_training_set should be
% arranged in a matrix of 35 rows and 70 columns to train the network.
% Remember that this arrangement should be the other way round (35 columns and 70 rows)
% for the SVM and Tree classifiers
input_training_set=input_training_set'; % get the right arrangement for the input set. 
output_training_set=output_training_set'; % get the right arrangement for the output set

Target=strcmp('female',output_training_set); % Now, Target has logical values (0 and 1). This has to be converted into double.

%*********************************************************************

num_features = 35; 
learning_rate = 0.1;
num_epochs = 100; 

% Linear Perceptron
nets = perceptron;
nets = train(nets, input_training_set, double(Target));


% Initialize and train the Linear SVM classifier
svm = fitcsvm(input_training_set', double(Target), 'KernelFunction', 'linear');
% Train the Classification Tree
tree = fitctree(input_training_set', double(Target), 'MaxNumSplits', 10);


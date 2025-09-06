% Project 1 - Amos Viray 23729527 Cameron Waddingham 23737222


%Initialising data and labels
features_def = [];
features_sob = [];
labels = [];

%Image acquisation and processing
human = [dir(fullfile('positive', '*.jpg')); dir(fullfile('positive', '*.png'))];
for i = 1:length(human) 
    filename = fullfile('positive', human(i).name);
    img = imresize(imread(filename),[128 64]);
    img = double(im2gray(img));
    feat_def = computeHOG(img);
    feat_sob = computeHOG_sobel(img);
    features_sob = [features_sob; feat_sob'];
    features_def = [features_def; feat_def'];
    labels = [labels; 1];
    
end

nonhuman = [dir(fullfile('negative', '*.jpg')); dir(fullfile('negative', '*.png'))];
for i = 1:length(nonhuman)
    filename = fullfile('negative', nonhuman(i).name);
    img = imresize(imread(filename),[128 64]);
    img = double(im2gray(img));
    feat_def = computeHOG(img);
    feat_sob = computeHOG_sobel(img);
    features_sob = [features_sob; feat_sob'];
    features_def = [features_def; feat_def'];
    labels = [labels; 0]; 
end

svmodel = fitcsvm(features_def, labels, 'KernelFunction', 'linear');
svmodel_sob = fitcsvm(features_sob, labels,'KernelFunction', 'linear');
%Extracts HOG features of images while satisfying requirements specified in
%the rubric
function hogfeature = computeHOG(img)
    %Defining parameters
    numbins = 9;
    binsize = 20;
    cellsize = 8;
    epsilon = 1e-5;
    blocksize = 2;

    %Filter without smoothing
    Gx = imfilter(img,[-1,0,1],"same"); 
    Gy = imfilter(img,[-1;0;1],"same");

    % Ablation
    % Gx = imfilter(img,fspecial('sobel'),"same"); 
    % Gy = imfilter(img,fspecial('sobel')',"same");

    %Gradient and angle calculations
    G = sqrt(Gx.^2 + Gy.^2);
    angle = atan2d(Gy, Gx);
    angle(angle < 0) = angle(angle < 0) + 180;
    
    %Bin storage
    bindex = floor(angle/binsize) + 1;
    bindex(bindex > numbins) = numbins;
    
    %Image dimensions
    [rows, cols] = size(img);
    numcellsy = floor(rows/cellsize);
    numcellsx = floor(cols/cellsize);
    
    %Histogram
    hist = zeros(numcellsy,numcellsx,numbins);
    for y = 1:numcellsy
        for x = 1:numcellsx
            rowstart = (y-1)*cellsize+1;
            colstart = (x-1)*cellsize+1;
            magblock = G(rowstart:rowstart+cellsize-1, colstart:colstart+cellsize-1);
            binblock = bindex(rowstart:rowstart+cellsize-1, colstart:colstart+cellsize-1);
    
            for i = 1:cellsize
                for j = 1:cellsize
                    b = binblock(i,j);
                    hist(y,x,b) = hist(y,x,b) + magblock(i,j);
                end
            end
        end
    end
    
    
    %Block normalisation
    hogfeature = [];
    for y = 1:numcellsy-blocksize+1
        for x = 1:numcellsx-blocksize+1
            block = hist(y:y+1, x:x+1, :);
            blockvect = reshape(block, [], 1);
            
            %L2 normalisation
            normblock = blockvect/sqrt(sum(blockvect.^2) + epsilon^2);
            
            %L2-hys clipping
            normblock(normblock > 0.2) = 0.2;
    
            %Renormalisation
            normblock = normblock/sqrt(sum(normblock.^2) + epsilon^2);
            
            %Append to HOG
            hogfeature = [hogfeature; normblock];
        end
    end
end

%Ablation study - Change [-1,0,1] filter to Sobel:
function hogfeature = computeHOG_sobel(img)
    %Defining parameters
    numbins = 9;
    binsize = 20;
    cellsize = 8;
    epsilon = 1e-5;
    blocksize = 2;

    % Ablation
    Gx = imfilter(img,fspecial('sobel'),"same"); 
    Gy = imfilter(img,fspecial('sobel')',"same");

    %Gradient and angle calculations
    G = sqrt(Gx.^2 + Gy.^2);
    angle = atan2d(Gy, Gx);
    angle(angle < 0) = angle(angle < 0) + 180;
    
    %Bin storage
    bindex = floor(angle/binsize) + 1;
    bindex(bindex > numbins) = numbins;
    
    %Image dimensions
    [rows, cols] = size(img);
    numcellsy = floor(rows/cellsize);
    numcellsx = floor(cols/cellsize);
    
    %Histogram
    hist = zeros(numcellsy,numcellsx,numbins);
    for y = 1:numcellsy
        for x = 1:numcellsx
            rowstart = (y-1)*cellsize+1;
            colstart = (x-1)*cellsize+1;
            magblock = G(rowstart:rowstart+cellsize-1, colstart:colstart+cellsize-1);
            binblock = bindex(rowstart:rowstart+cellsize-1, colstart:colstart+cellsize-1);
    
            for i = 1:cellsize
                for j = 1:cellsize
                    b = binblock(i,j);
                    hist(y,x,b) = hist(y,x,b) + magblock(i,j);
                end
            end
        end
    end
    
    
    %Block normalisation
    hogfeature = [];
    for y = 1:numcellsy-blocksize+1
        for x = 1:numcellsx-blocksize+1
            block = hist(y:y+1, x:x+1, :);
            blockvect = reshape(block, [], 1);
            
            %L2 normalisation
            normblock = blockvect/sqrt(sum(blockvect.^2) + epsilon^2);
            
            %L2-hys clipping
            normblock(normblock > 0.2) = 0.2;
    
            %Renormalisation
            normblock = normblock/sqrt(sum(normblock.^2) + epsilon^2);
            
            %Append to HOG
            hogfeature = [hogfeature; normblock];
        end
    end
end

%Testing SVM:
testfeatures = [];
testfeatures_sob = [];
truelabels = [];

testpos = dir(fullfile('test_pos','*.png'));
for i = 1:length(testpos)
    filename = fullfile('test_pos',testpos(i).name);
    img = imresize(imread(filename),[128 64]);
    img = double(im2gray(img));
    feat_def = computeHOG(img);
    feat_sob = computeHOG_sobel(img);
    testfeatures_sob = [testfeatures_sob; feat_sob'];
    testfeatures = [testfeatures; feat_def'];
    truelabels = [truelabels; 1];
end

testneg = dir(fullfile('test_neg','*.png'));
for i = 1:length(testneg)
    filename = fullfile('test_neg',testneg(i).name);
    img = imresize(imread(filename),[128 64]);
    img = double(im2gray(img));
    feat_def = computeHOG(img);
    feat_sob = computeHOG_sobel(img);
    testfeatures_sob = [testfeatures_sob; feat_sob'];
    testfeatures = [testfeatures; feat_def'];
    truelabels = [truelabels; 0];
end

% Miss rate vs FPPW
[~,scores_def] = predict(svmodel, testfeatures);
[~,scores_sob] = predict(svmodel_sob, testfeatures_sob);
% Get SVM scores
scores1 = scores_def(:,2); % Confidence for positive class
scores2 = scores_sob(:,2);
% Define thresholds sweeping range based on min and max scores
thresholds = linspace(min([scores1; scores2]), max([scores1; scores2]), 100);
% Preallocate
missrate_def = zeros(size(thresholds));
fppw_def = zeros(size(thresholds));
missrate_sob = zeros(size(thresholds));
fppw_sob = zeros(size(thresholds));
% Total number of test windows
N = length(truelabels);
% Sweep thresholds
for i = 1:length(thresholds)
t = thresholds(i);
% Default Filter metrics:
preds = scores1 >= t;
tp = sum((preds == 1) & (truelabels == 1));
fp = sum((preds == 1) & (truelabels == 0));
fn = sum((preds == 0) & (truelabels == 1));
missrate_def(i) = fn / (tp + fn);
fppw_def(i) = fp / N;
% Sobel Filter metrics:
preds = scores2 >= t;
tp = sum((preds == 1) & (truelabels == 1));
fp = sum((preds == 1) & (truelabels == 0));
fn = sum((preds == 0) & (truelabels == 1));
missrate_sob(i) = fn / (tp + fn);
fppw_sob(i) = fp / N;
end
% Plot Miss Rate vs FPPW
figure;
semilogx(fppw_def, missrate_def, 'b-', 'LineWidth', 2); hold on;
semilogx(fppw_sob, missrate_sob, 'r--', 'LineWidth', 2);
xlabel('False Positives Per Window (FPPW)');
ylabel('Miss Rate');
title('Miss Rate vs FPPW Curve');
legend('Default: [-1,0,1]', 'Sobel');
grid on;

%TP Rate vs FP Rate - ROC Curve:
%Plot
[xdef,ydef,~,auc_def] = perfcurve(truelabels, scores1,1);
[xsob,ysob,~,auc_sob] = perfcurve(truelabels, scores2,1);

figure;
plot(xdef, ydef, 'b-', 'LineWidth', 2);
hold on;
plot(xsob, ysob, 'r--', 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend(sprintf('Default HOG (AUC=%.2f)', auc_def), sprintf('Sobel HOG (AUC=%.2f)', auc_sob));
title('True Positive Rate vs False Positive Rate');
grid on;

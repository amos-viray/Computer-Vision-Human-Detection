% Project 1 - Amos Viray 23729527 Cameron Waddingham 23737222


%Initialising data and labels
features = [];
labels = [];

%Image acquisation and processing
human = [dir(fullfile('positive', '*.jpg')); dir(fullfile('positive', '*.png'))];
for i = 1:length(human)
    filename = fullfile('positive', human(i).name);
    img = imresize(imread(filename),[128 64]);
    img = double(im2gray(img));
    feat = computeHOG(img);
    features = [features; feat']; %Transform column vector to row vector
    labels = [labels; 1];
end

nonhuman = [dir(fullfile('negative', '*.jpg')); dir(fullfile('negative', '*.png'))];
for i = 1:length(nonhuman)
    filename = fullfile('negative', nonhuman(i).name);
    img = imresize(imread(filename),[128 64]);
    img = double(im2gray(img));
    feat = computeHOG(img);
    features = [features; feat']; %Transform column vector to row vector
    labels = [labels; 0]; 
end

svmodel = fitcsvm(features, labels, 'KernelFunction', 'linear');
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
    Gx = imfilter(img,[-1,0,1],"same");%Horizontal gradient
    Gy = imfilter(img,[-1;0;1],"same");%Vertical gradient

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
    hist = zeros(numcellsy,numcellsx,numbins); %Generate 3D array to store histograms
    %Iterate through each cell in the grid
    for y = 1:numcellsy
        for x = 1:numcellsx
            %Calculate starting pixel position of cell in original image
            rowstart = (y-1)*cellsize+1;
            colstart = (x-1)*cellsize+1;
            magblock = G(rowstart:rowstart+cellsize-1, colstart:colstart+cellsize-1); %Extract gradient magnitude of cell
            binblock = bindex(rowstart:rowstart+cellsize-1, colstart:colstart+cellsize-1); %Extract bin index of cell
    
            for i = 1:cellsize
                for j = 1:cellsize
                    b = binblock(i,j); %Find bin index of pixel's orientation
                    hist(y,x,b) = hist(y,x,b) + magblock(i,j); %Append gradient magnitude to corresponding bin in histogram
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
truelabels = [];

%File extraction
testpos = dir(fullfile('test_pos','*.png'));
for i = 1:length(testpos)
    filename = fullfile('test_pos',testpos(i).name);
    img = imresize(imread(filename),[128 64]);
    img = double(im2gray(img));
    feat = computeHOG(img);
    testfeatures = [testfeatures; feat']; 
    truelabels = [truelabels; 1]; %Generate ground labels
end

testneg = dir(fullfile('test_neg','*.png'));
for i = 1:length(testneg)
    filename = fullfile('test_neg',testneg(i).name);
    img = imresize(imread(filename),[128 64]);
    img = double(im2gray(img));
    feat = computeHOG(img);
    testfeatures = [testfeatures; feat'];
    truelabels = [truelabels; 0];
end

predictedlabels = predict(svmodel, testfeatures);

%Confusion Matrix:
tp = sum((predictedlabels == 1) & (truelabels == 1));
tn = sum((predictedlabels == 0) & (truelabels == 0));
fp = sum((predictedlabels == 1) & (truelabels == 0));
fn = sum((predictedlabels == 0) & (truelabels == 1));

%Metrics:
accuracy = (tp+tn)/(tp+tn+fp+fn);
precision = tp/(tp+fp);
recall = tp/(tp+fn);
missrate = fn/(tp+fn);
fppw = fp/(tp+tn+fp+fn);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f%%\n', precision * 100);
fprintf('Recall: %.2f%%\n', recall * 100);
fprintf('Miss Rate: %.2f%%\n', missrate * 100);
fprintf('FPPW: %.4f\n', fppw);

save('svmodel.mat', 'svmodel');

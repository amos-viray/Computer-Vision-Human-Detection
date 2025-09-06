function hogfeature = computeHOG(img)
    %Defining parameters
    numbins = 9;
    binsize = 20;
    cellsize = 8;
    epsilon = 1e-5;
    blocksize = 2;
    
    img = imresize(img, [128 64]);
    img = double(im2gray(img));

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
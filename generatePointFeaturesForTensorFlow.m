function [features, labels] = ...
    generatePointFeaturesForTensorFlow(points, voxelSizeInMeters, pointClassesOfInterest)

rx = max(points(:,1)) - min(points(:,1));
ry = max(points(:,2)) - min(points(:,2));
rz = max(points(:,3)) - min(points(:,3));

nVoxels = [ceil(rx / voxelSizeInMeters), ...
            ceil(ry / voxelSizeInMeters), ...
            ceil(rz / voxelSizeInMeters)];

%Adding i,j,k coords.
pointVoxelIndices = [floor(normalize0To1(points(:,1)) *nVoxels(1)) ...
                      floor(normalize0To1(points(:,2)) * nVoxels(2)) ...
                      floor(normalize0To1(points(:,3)) * nVoxels(3))];
pointVoxelIndices(pointVoxelIndices == 0) = 1;

%Calculating linear indices
pointVoxelID = sub2ind(nVoxels, pointVoxelIndices(:,1), pointVoxelIndices(:,2), pointVoxelIndices(:,3));
pointPixelID = sub2ind(nVoxels, pointVoxelIndices(:,1), pointVoxelIndices(:,2));

pointGroundHeights = inf(length(pointPixelID), 1); %Ground H.
pointMaxColumnHeights = - inf(length(pointPixelID), 1); %Max H.

%Reordering all points by voxelID
[~,I] = sort(pointVoxelID);
points = points(I, :);
pointVoxelID = pointVoxelID(I);
%pointVoxelIndices = pointVoxelIndices(I,:);

[f,~] = analyzeVoxel(points(1,:),[]);
nFeatures = length(f);
voxelFeatures = zeros(prod(nVoxels), nFeatures); %NFeatures will grow only once
voxelClass = zeros(prod(nVoxels), 1);

i = 1;
while i <= length(pointVoxelID)
    ini = i;
    
    while i <= length(pointVoxelID) && pointVoxelID(ini) == pointVoxelID(i)
        i = i+1;
    end
    ps = points(ini: i-1, :);
    
    voxelID = pointVoxelID(ini);
    pixelID = pointPixelID(ini);
    minH = min([ pointGroundHeights(pixelID) ps(:,3)']);
    pointGroundHeights(ini: i-1) = minH;
    
    pointMaxColumnHeights(ini: i-1) = max([ pointMaxColumnHeights(pixelID) ps(:,3)']);
    
    [f, c] = analyzeVoxel(ps, pointClassesOfInterest);
    voxelFeatures(voxelID,:) = f;
    voxelClass(voxelID) = c;
end

%Adjusting height with area surface elevation
for i = 1:length(pointGroundHeights)
    voxelID = pointVoxelID(i);
    voxelFeatures(voxelID, 1:3) = voxelFeatures(voxelID, 1:3) - pointGroundHeights(i);
end

% %Extending features with surrounding voxels
if (1)
    extendedVoxelFeatures = extendFeaturesWithNeighbourgs(voxelFeatures, nVoxels); 
    voxelFeatures = extendedVoxelFeatures;
end

features = [points(:,1), points(:,2), points(:,3), points(:,4), ...
            points(:,3) - pointGroundHeights, ...
            pointMaxColumnHeights - pointGroundHeights]; %Column height
features = [features, voxelFeatures(pointVoxelID, :)];

labels = ismember(points(:, 5), pointClassesOfInterest);

end

function [features, class] = analyzeVoxel(ps, pointClassesOfInterest)
    features = [minMaxMean(ps(:,3)), ... % Z
                           minMaxMean(ps(:,4)), ... %Intensity
                           minMaxMean(ps(:,6)), ...%Return value
                           size(ps, 1)]; %N Points  
    class = sum(ismember(ps(:, 5), pointClassesOfInterest)) > 0;
end


function r = minMaxMean(v)
    r = [min(v), max(v), mean(v)];
end

function v2 = normalize0To1(v)
    v2 = (v - min(v)) / (max(v) - min(v));
end

function extendedVoxelFeatures = extendFeaturesWithNeighbourgs(voxelFeatures, nVoxels)

    originalIndices = 1:length(voxelFeatures);
    right = originalIndices;
    left = originalIndices;
    forward = originalIndices;
    rear = originalIndices;
    bottom = originalIndices;
    top = originalIndices;
    [i, j, k] = ind2sub(nVoxels, originalIndices);
    
    %Forward
    noForwardLimit = i< nVoxels(1);
    forward(noForwardLimit) = sub2ind(nVoxels, i(noForwardLimit)+1, j(noForwardLimit), k(noForwardLimit));
    
    %Rear
    noRearLimit = i>1;
    rear(noRearLimit) = sub2ind(nVoxels, i(noRearLimit)-1, j(noRearLimit), k(noRearLimit));
    
    %Right
    noRightLimit = j< nVoxels(2);
    right(noRightLimit) = sub2ind(nVoxels, i(noRightLimit), j(noRightLimit)+1, k(noRightLimit));
    
    %Left
    noLeftLimit = j>1;
    left(noLeftLimit) = sub2ind(nVoxels, i(noLeftLimit), j(noLeftLimit)-1, k(noLeftLimit));
    
    %Top
    noTopLimit = k< nVoxels(3);
    top(noTopLimit) = sub2ind(nVoxels, i(noTopLimit), j(noTopLimit), k(noTopLimit)+1);
    
    %Bottom
    noBottomLimit = k>1;
    bottom(noBottomLimit) = sub2ind(nVoxels, i(noBottomLimit), j(noBottomLimit), k(noBottomLimit)-1);
    
    feats = [3,6,9,10]; %Mean values + nPoints
    
    extendedVoxelFeatures = [voxelFeatures ...
                            voxelFeatures(forward, feats) ...
                             voxelFeatures(rear, feats) ...
                             voxelFeatures(right, feats) ...
                             voxelFeatures(left, feats) ...
                             voxelFeatures(top, feats) ...
                             voxelFeatures(bottom, feats) ...
                             ];
end

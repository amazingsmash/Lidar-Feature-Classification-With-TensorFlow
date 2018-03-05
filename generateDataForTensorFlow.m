
i = 26;
voxelSizeInMeters = 2.0;
pointClassesOfInterest = [16];

samples = [];

for i = 26:28
    filename = sprintf('DATASET/%06d-expected.mat', i);
    load(filename)
    fprintf('Loaded %s\n', filename);

%     [points, pointVoxelID, pointGroundHeights, voxelFeatures, ~, nVoxels] = ...
%             pointsToVoxels(points, voxelSize, pointClassesOfInterest);
% 
%     features = [points(:,1), points(:,2), ...
%                 points(:,3), points(:,4), points(:,3) - pointGroundHeights];
%     features = [features, voxelFeatures(pointVoxelID, :)];
% 
%     labels = points(:,5) == 16; %Turrets

[features, labels] = ...
    generatePointFeaturesForTensorFlow(points, voxelSizeInMeters, pointClassesOfInterest);

    samplesFile = [features labels];
    samples = [samples; samplesFile];
end

samples(:,1) = samples(:,1) - min(samples(:,1));
samples(:,2) = samples(:,2) - min(samples(:,2));
samples(:,3) = samples(:,3) - min(samples(:,3));

%Balance classes
ratio = sum(labels) / sum(~labels);
ind = labels | rand(length(labels), 1) < ratio; 
balancedSamples = samples(ind, :);

%Shuffle
balancedSamples = balancedSamples(randperm(length(balancedSamples)), :);

%Train and test
trainRatio = 0.9;
cut = round(trainRatio * length(balancedSamples));
trainSamples = balancedSamples(1:cut, :);
testSamples = balancedSamples(cut+1:end, :);

save('TF_Data.mat', 'samples', 'trainSamples', 'testSamples');



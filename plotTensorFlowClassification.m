%Plot tensorflow generated classification
figure;
hold on;
l = pointsPredicted(:,4) == 1;


plot3(pointsPredicted(l,1), pointsPredicted(l,2), pointsPredicted(l,3), '.r')
plot3(pointsPredicted(~l,1), pointsPredicted(~l,2), pointsPredicted(~l,3), '.b')
 
%color = [l, zeros(length(l),1), ~l];
%scatter3(pointsPredicted(:,1), pointsPredicted(:,2), pointsPredicted(:,3), 1, l)

%read data example: Import columns as column vectors 
XY = csvread('/Users/santiago/eth/21a/PAI/PAI_projects/task1_handout_d3d63876/train_x.csv', 1,0);
z = csvread('/Users/santiago/eth/21a/PAI/PAI_projects/task1_handout_d3d63876/train_y.csv', 1,0);
x = XY(:,1);
y = XY(:,2);

% make grid and interpolate
%[xi,yi] = meshgrid(0:0.01:0.9988, 0:0.01:0.9988);
%zi = griddata(x,y,z,xi,yi);

% plot
%surf(xi, yi, zi);
subplot(2,1,1);
scatter3(x, y, z, [], z, 'filled');


% Interpolated data
XY_2 = csvread('/Users/santiago/eth/21a/PAI/PAI_projects/task1_handout_d3d63876/interpolated_x_2500.csv', 1,0);
z_2 = csvread('/Users/santiago/eth/21a/PAI/PAI_projects/task1_handout_d3d63876/interpolated_y_2500.csv', 1,0);
x_2 = XY_2(:,1);
y_2 = XY_2(:,2);

subplot(2,1,2); 
scatter3(x_2, y_2, z_2, [], z_2, 'filled');
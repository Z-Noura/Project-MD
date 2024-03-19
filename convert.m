clear, close all;

%% load the sample data
mesh = stlread('Femur.stl');


% first, generate a surface from the original image
% similar to demo_shortcuts_ex1.m

[img, v2smap]=s2v(mesh.Points,mesh.ConnectivityList,50);
save('femur.mat','img');
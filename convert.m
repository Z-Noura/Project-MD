clear, close all;

%% load the sample data
mesh = stlread('MarqueurCircles.stl');


% first, generate a surface from the original image
% similar to demo_shortcuts_ex1.m

[img, v2smap]=s2v(mesh.Points,mesh.ConnectivityList,250);
save('Cercles.mat','img');
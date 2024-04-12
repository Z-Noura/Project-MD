clear, close all;

%% load the sample data
mesh = stlread('Marqueur 2 v8.stl');


% first, generate a surface from the original image
% similar to demo_shortcuts_ex1.m

[img, v2smap]=s2v(mesh.Points,mesh.ConnectivityList,50);
save('BcpDeCercles.mat','img');
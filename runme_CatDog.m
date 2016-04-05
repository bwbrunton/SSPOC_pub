% SSPOC Example: Cat v Dog
clear all; close all;

data_dir = './data/';

load([data_dir 'CatDog.mat']);
% the CatDog.mat data file contains the following variables:
%
% X     data matrix, 4096 (nx by ny) pixels by 242 images
%
% G     vector of categories, 0 = cat and 1 = dog
%
% nx    number of horizontal pixels in each image
%
% ny    number of vertical pixels in each image

n = nx * ny;
classes = unique(G);
c = numel(classes); % number of groups

%% take a look at the singular value spectrum

[U, Sigma, V] = svd(X, 'econ');
dS = diag(Sigma)/sum(Sigma(:));

figure;
semilogy(dS(1:235), 'k');
xlabel('i');
ylabel('\sigma_i');
grid on;
title('Singular Value Spectrum');

set(gcf, 'Position', [100 100 300 300]);

%% compute PCA-LDA and SSPOC for X and G

r = 20; % truncate to first r features
[w, Psi] = PCA_LDA(X, G, 'nFeatures', r);

s = SSPOC(Psi, w);

%% construct sparse measurement matrix
% pick non-zero elements of s: these are the sensor locations
sensors = find(abs(s) >= norm(s, 'fro')/c/r/2);
q = numel(sensors);

% construct the measurement matrix Phi
Phi = zeros(q, n);
for qi = 1:q,
    Phi(qi, sensors(qi)) = 1;
end;

%% learn new classifier for sparsely measured data

w_sspoc= LDA_n(Phi * X, G);

Xcls = w_sspoc' * (Phi * X);

% compute centroid of each class in classifier space
centroid = zeros(c-1, c);
for i = 1:c,
    centroid(:,i) = mean(Xcls(:,G==classes(i)), 2);
end;

% use sparse sensors to classify X
cls = classify_nc(X, Phi, w_sspoc, centroid);

% print classification accuracy
fprintf('Classification accuracy (not cv) of %i SSPOC sensors is %g.\n',...
    q, sum(cls-1 == G)/numel(Xcls));

%% visualive Psi*w and s (Fig 2)
chi = -Psi * w;
sensors_x = rem(sensors, ny);
sensors_y = ceil(sensors/nx);

figure;
subplot(1, 2, 1);
hold on;
plot(chi/max(abs(chi)), 'Color', 0.5*[1 1 1]);
plot(s/max(abs(s)), 'r', 'LineWidth', 2);
axis tight;
axis square;

subplot(1, 2, 2);
hold on;
imagesc(reshape(chi, nx, ny));
colormap gray;
axis ij;
plot(sensors_y, sensors_x, 'r.', 'MarkerSize', 15);
axis square;
axis off;
set(gcf, 'Position', [100 100 800 400]);
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0 0 6 3], ...
    'PaperPositionMode', 'manual');



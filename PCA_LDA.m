function [w, Psi, centroid] = PCA_LDA(X, G, varargin)
% function [w, Psi, centroid] = PCA_LDA(X, G, varargin)
%
%
% INPUTS:
%
% X         
%           a [n x m] matrix of the raw data, where there are n samples with m
%           measurements each
%
% G     
%           a [1 x m] vector with the group id's of each sample in X; it's
%           nice when they're 0's and 1's, etc.
%
% VARIABLE PARAMETERS
%
% nFeatures 
%           number of features used in the discrimination
%
% OUTPUTS:
%
% w
%           the LDA vectors learned in the classification using nFeatures
%           a [nFeatures x c-1] matrix
%
% Psi
%           the feature basis used in the classification
%
% centroid
%           the centroid of each class in w space
%           a [nFeatures x c] matrix
%           useful for nearest centroid classification
%
% BWB, July 2015

% input parsing
p = inputParser; 

% required inputs
p.addRequired('X', @isnumeric);
p.addRequired('G', @(x)length(x)==size(X,2));

% parameter value iputs
p.addParameter('nFeatures', 10, @(x)isnumeric(x) && x<=size(X,2));

% now parse the inputs
p.parse(X, G, varargin{:});
inputs = p.Results;

classes = unique(G);
c = numel(classes); % number of groups


% compute feature basis Psi
[U, ~, ~] = svd(X, 0);
Psi = U(:, 1:inputs.nFeatures);

a = Psi'*X;

% LDA
w = LDA_n(a, G);

Xcls = w' * a;

% compute centroid of each class in classifier space
centroid = zeros(c-1, c);
for i = 1:c,
    centroid(:,i) = mean(Xcls(:,G==classes(i)), 2);
end;


function [s] = SSPOC(Psi, w, varargin)
% function [s] = SSPOC(Psi, w, varargin)
%
% INPUTS:
%
% Psi
%           feature space
%           for example, from SVD of the original data
%           n by r matrix, n is number of measurements, r is number of
%           features
%
% w
%           subspace in Psi that we want to reconstruct
%           r by c matrix, r must match Psi, c is the number of clusters
%           minus 1 (the dimensionality of the decision space)
%
% VARARGIN:
%
% lambda
%           the coupling term between w vectors
%           if lambda = 0 (default), there is no coupling 
%           if lambda > 0, then there is increasing overlap between the
%           sparse solutions s that recover w 
%           (lambda can be tuned between 0.1 and 100 on log scale)
%
% epsilon
%           error tolerance for reconstruction
%           epsilon 
%
% OUTPUTS:
%
% s
%           sparse vector in space of the data;
%           non-zero entries in s correspond to sparse sensor locations
%
% BWB, June 2015

% input parsing
p = inputParser; 

% required inputs
p.addRequired('Psi', @isnumeric);
p.addRequired('w', @isnumeric);

% parameter value iputs
p.addParameter('lambda', 0, @(x)isnumeric(x) && x>=0);
p.addParameter('epsilon', 1e-10, @(x)isnumeric(x) && x>=0);

% now parse the inputs
p.parse(Psi, w, varargin{:});
inputs = p.Results;


% define some dimensions
[n, r] = size(Psi); %#ok<ASGLU>
c = size(w, 2);

% solve Psi'*s = w
% using the l1 norm to promote sparsity
unit = ones(c,1);
cvx_begin quiet
    variable s( n, c );
    minimize( norm(s(:),1) + inputs.lambda*norm(s*unit,1) ); %#ok<NODEF>
    subject to
        norm(Psi'*s - w, 'fro') <= inputs.epsilon; %#ok<VUNUS>
cvx_end
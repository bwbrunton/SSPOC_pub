function [w, D] = LDA_n(X, G)
% function [w, D] = LDA_n(X, G)
%
% n-way LDA
% returns solutions of the generalized eigenvector problem maximizing
% between-class variance and minimizing within-class variance
%
% BWB, June 2015

d = size(X, 1);
classes = unique(G);
c = numel(classes);

N = zeros(1, c);
centroid = zeros(d, c);
for i = 1:c,
    N(i) = sum(G==classes(i));
    centroid(:,i) = mean(X(:, G==classes(i)), 2);
end;

% the within class variance
Sw = zeros(d, d);
for i = 1:c,
    res = X(:,G==classes(i)) - centroid(:,i)*ones(1,N(i));
    Sw = Sw + (res)*(res)';
end;

% the between class variance
Sb = zeros(d, d);
for i = 1:c,
    Sb = Sb + N(i) * (centroid(:,i)-mean(X,2))*(centroid(:,i)-mean(X,2))';
end;

% solve for the eigenvalues of inv(Sw)*Sb,
% keep eigenvectors with c-1 largest magnitude eigenvalues
[w, D] = eigs(pinv(Sw) * Sb, c-1);

% normalize w vectors
for i = 1:size(w,2), 
    w(:,i) = w(:,i) / sqrt(w(:,i)'*w(:,i));
end;
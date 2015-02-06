function [ subSpace ] = construct_subspace(Y,nn)
n = size(Y, 2);
M = repmat(mean(Y, 2), 1, n);
C = ((Y - M) * (Y - M)') ./ (n - 1);
lamda = 0.001 * trace(C);
C = C + lamda * eye(size(C, 1));

[u,~,~] = svd(C);
subSpace = u(:,1:nn);
end



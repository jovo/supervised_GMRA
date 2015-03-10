for i= 5:2:14;
    A = magic(i);
    B = bsxfun(@minus, A, mean(A,2));
    [V, S, W] = randPCA(B, 4);
    [coef,score,latent] = pca(B);
    diag(S);
    diag(S)/sqrt(i)
end

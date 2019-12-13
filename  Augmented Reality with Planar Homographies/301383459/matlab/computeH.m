function [H2to1] = computeH(matches)

    A(:,1) = matches(:,3) .* matches(:,1);
    A(:,2) = matches(:,3) .* matches(:,2);
    A(:,3) = matches(:,3);
    A(:,4) = matches(:,4) .* matches(:,1);
    A(:,5) = matches(:,4) .* matches(:,2);
    A(:,6) = matches(:,4);
    A(:,7) = matches(:,1);
    A(:,8) = matches(:,2);
    A(:,9) = 1;
    
    [U, S, V] = svd(A);
    H2to1 = V(:, end);
    H2to1 = reshape(H2to1, [3 3]);
    H2to1 = H2to1';

    [Uf, Sf, Vf] = svd(H2to1);
    Sf(3,3) = 0;
    H2to1 = Uf * Sf * Vf';
end
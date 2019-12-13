function [H2to1] = computeH_norm(matches)
    x1 = mean(matches(:,1));
    y1 = mean(matches(:,2));
    x1dash = var(matches(:,1));
    y1dash = var(matches(:,2));
    x2 = mean(matches(:,3));
    y2 = mean(matches(:,4));
    x2dash = var(matches(:,3));
    y2dash = var(matches(:,4));

    T1 = zeros(3, 3);
    T1 = [1/x1dash 0 -x1/x1dash; 0 1/y1dash -y1/y1dash;0 0 1];
    X1 = [matches(:,1:2), ones(size(matches,1),1)];
    X1dash = T1 * X1';
    X1dash = X1dash';
    
    T2 = [1/x2dash 0 -x2/x1dash; 0 1/y2dash -y2/y2dash;0 0 1];
    X2 = [matches(:,3:4), ones(size(matches,1),1)];
    X2dash = T2 * X2';
    X2dash = X2dash';
    
    norm_coor = [X1dash(:,1:2), X2dash(:,1:2)];
    H2to1 = computeH(norm_coor);
    H2to1 = T2' * H2to1 * T1;
    H2to1 = H2to1/H2to1(3,3);
end
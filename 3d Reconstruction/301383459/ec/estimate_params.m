function [K, R, t] = estimate_params(P)
% ESTIMATE_PARAMS computes the intrinsic K, rotation R and translation t from
% given camera matrix P.

 Ro = P(:,1:3);
 t1 = P(:,4);
 
  % start performing QR decomposition
  reverse_rows = [0, 0, 1; 
                  0, 1, 0; 
                  1, 0, 0];
  [Q1, R1] = QR((reverse_rows * Ro)');
  
  
  K = reverse_rows * R1' * reverse_rows;
  R = reverse_rows * Q1';
  t = inv(K) * t1;

end
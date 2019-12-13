function P = estimate_pose(x1, X1)
A = [];
x1=x1';
X1=X1';

  for i=1:size(x1, 1)
    % image plane coord.
    x = x1(i, 1);
    y = x1(i, 2);
    % corresponding world coord.
    X = X1(i, 1);
    Y = X1(i, 2);
    Z = X1(i, 3);
    A = [...
      A; ...
      X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x; ...
      0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y  ...
    ];
  end
  % compute eigenvector corresponding smallest eigenvalue from A'A to get camera projection matrix
  ATA = A'*A;
  [SmallEigVector, SmallEigValue] = eigs(ATA, 1, 'SM');
  P = [ ...
        SmallEigVector(1), SmallEigVector(2), SmallEigVector(3), SmallEigVector(4);   ...
        SmallEigVector(5), SmallEigVector(6), SmallEigVector(7), SmallEigVector(8);   ...
        SmallEigVector(9), SmallEigVector(10), SmallEigVector(11), SmallEigVector(12) ...
  ];
end
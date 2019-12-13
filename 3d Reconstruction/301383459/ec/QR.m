function [Q, R] = QR(A)
  
  v1 = A(:,1);
  v2 = A(:,2);
  v3 = A(:,3);
  
  u1 = v1;
  u2 = v2 - projection(u1, v2);
  u3 = v3 - projection(u1, v3) - projection(u2, v3);
  
  
  Q = zeros(3,3);
  Q(:,1) = u1/norm(u1);
  Q(:,2) = u2/norm(u2);
  Q(:,3) = u3/norm(u3);
 
  R = Q'*A;
end

function projection_uv = projection(u, v)
  projection_uv = (dot(u, v)/dot(u, u))*u;
end
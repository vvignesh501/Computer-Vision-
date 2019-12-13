function [ K ] = get_kernel(input_data, train_data, p)
%GET_KERNEL
% Computes the kernel

    M = length(input_data(1,:));
    N = length(train_data(1,:));
    K = zeros(M, N);
    for i = 1:M
        for j = 1:N
            if p == 0 % linear kernel
                K(i,j) = input_data(:,i)' * train_data(:,j);
            else % polynomial kernel
                K(i,j) = (input_data(:,i)' * train_data(:,j) + 1)^p;
            end
        end
    end

end


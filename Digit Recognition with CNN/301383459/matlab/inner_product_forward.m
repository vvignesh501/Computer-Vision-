function [output] = inner_product_forward(input, layer, param)

% set the shape of output
output.height = 1;
output.width = 1;
output.channel = layer.num;
output.batch_size = input.batch_size;

% sanity check
d = size(input.data, 1);
assert(size(param.w, 1) == d, 'dimension mismatch in inner_product layer');

% initialize the outupt data
output.data = zeros(layer.num, input.batch_size);

% start to work here to compute output.data
dt = input.data ;
for i=1:input.batch_size
    dti=dt(:,i);
    output.data(:,i) = (dti'*param.w + param.b)' ;
    
end
end

function [param_grad, input_od] = inner_product_backward(output1, input, layer, param)

input_od = zeros(size(input.data));
param_grad.b = zeros(size(param.b));
param_grad.w = zeros(size(param.w));

param_grad.b = sum(output1.diff')  ;
param_grad.w = input.data*(output1.diff')  ;

input_od = param.w*output1.diff ;


end

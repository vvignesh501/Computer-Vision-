function [output] = pooling_layer_forward(input, layer)

% figure out the output shape
h_in = input.height;
w_in = input.width;
c = input.channel;
batch_size = input.batch_size;
k = layer.k;
layer.pad = 0;
pad = layer.pad;
stride = layer.stride;

h_out = (h_in + 2*pad - k) / stride + 1;
w_out = (w_in + 2*pad - k) / stride + 1;
assert(h_out == floor(h_out), 'h_out is not integer')
assert(w_out == floor(w_out), 'w_out is not integer')

% set output shape
output.height = h_out;
output.width = w_out;
output.channel = c;
output.batch_size = batch_size;

% initialize output.data
output.data = zeros(h_out*w_out*c, batch_size);
dt = input.data ;

% work out the average pooling and compute output.data
for i1=1:batch_size
    dti=reshape(dt(:,i1),h_in,w_in,c) ;
    for j1=1:c
        for k1=1:h_out
            for l1=1:w_out
                temp1=dti((k1-1)*stride+1:(k1-1)*stride+k,(l1-1)*stride+1:(l1-1)*stride+k,j1);
                outdt(k1,l1,j1)=max(temp1(:));
             end
         end
     end
     output.data(:,i1)= reshape(outdt,h_out*w_out*c,1) ;
end
        
end


clear
rand('seed', 100000)
randn('seed', 100000)

% lenet configuration

% first layer is data layer
layers{1}.type = 'DATA';
% define the input shape
layers{1}.height = 28;
layers{1}.width = 28;
layers{1}.channel = 1;
layers{1}.batch_size = 64;

layers{2}.type = 'CONV'; % second layer is conv layer
layers{2}.num = 20; % number of output channel
layers{2}.k = 5; % kernel size
layers{2}.stride = 1; % stride size
layers{2}.pad = 0; % padding size
layers{2}.group = 1; % group of input feature maps
                     % you can ignore this 

layers{3}.type = 'RELU'; % relu layer

layers{4}.type = 'LOSS'; % loss layer
layers{4}.num = 10; % number of classes (10 digits)


% load data
load_mnist_all

xtrain = [xtrain, xvalidate];
ytrain = [ytrain, yvalidate];
m_train = size(xtrain, 2);
ytest=ytest';
pred=ytest(64:127,1);
batch_size = 64;

% display information
test_interval = 500;
display_interval = 10;
snapshot = 5000;
max_iter = 10;

% initialize all parameters in each layers
params = init_convnet(layers);

for iter = 1 : max_iter
    % randomly fetch a batch
    id = randi([1 m_train], batch_size, 1);
    % forward and backward
    [output, P, indices] = conv_net(params, layers, xtrain(:, id), ytrain(id));
end

truth=ytest(1:64,1);
confusion=zeros(10,10);
for i=1:size(pred)
    confusion(pred(i),truth(i))=confusion(pred(i),truth(i))+1;
end

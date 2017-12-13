clear
filename = 'train.mat';
load(filename)
filename = 'test.mat';
load(filename)

w1 = [];
w1 = rand(785,30); % each row got weight values for separate xi
bias = [];
bias = rand(2,1); % bias value is same for the whole layer; we have two instances here so the first one is for hidden layer and the second one is for output
w2 = [];
w2 = rand(30,10); % no of rows, 	  of cols
hidden_layer_values = [];
hidden = [];

accuracy = 0;


%Apply softmax on target values and compute error
val = 0;
sum_val = 0;

for c = 1:10
    sum_val = sum_val + exp(c);
end

target_softmax = [];
for b = 1:10
    val = exp(b);  
    target_softmax(end+1) = val / sum_val;
    val =0;
end
target_softmax;
temp = [];
temp = w1;

w1(1,:) 
disp('displayed');
for q = 1:60000
x1= [];
x2 = [];
v1= [];
v2 =[];

summat = 0;
hidden_layer_values = [];
for b = 1:30
  v = [];
  v1 = transpose(w1(:,b));
  v2 = (train.X(:,q)); 
  v = v1 * v2;
 % for l = 1:785
 % summat = summat  + v(l);
 % end
  hidden_layer_values(end+1) =  1 ./ (1+ exp(-(v))); 
  summat = 0;
end
hidden_layer_values;
output_layer_values = [];
summat = 0;
for b = 1:10
     x1 = transpose(w2(:,b));
     x2 = transpose(hidden_layer_values(1,:));
     x = x1 * x2;
     %whos
     %for l = 1:30
     %summat = summat  + x(l);
     %end
     output_layer_values(end+1) =  x;
     summat = 0;
end
output_layer_values;
softmax_values = [];
ans = 0;
sum_ans = 0;

%denominator factor
for c = 1:10 %number of columns
    sum_ans = sum_ans + exp(output_layer_values(c));
end

for b = 1:10 %number of columns
    ans = exp(output_layer_values(b));
    softmax_values(end +1) = ans / sum_ans;
    ans = 0;
end
softmax_values;

% computing error
error_values = [];
v = [];
v = softmax_values - target_softmax;
y = v.^2; %squaring each value
error_values = y * 0.5;
%total_error = sum (error_values)
%er = total_error;           
% now do backward propagation
learning_rate = 0.3;
vect = [];
vect = (target_softmax - softmax_values);
 weight_vector = [];

%error term for output layer
 for b = 1:10
     weight_vector(end+1) = vect(b) * softmax_values(b) *(1-softmax_values(b));
 end


w3 = w2;
%weight update
for a = 1:30
     for b = 1:10
     w2(a,b) = w2(a,b) + (weight_vector(b) * hidden_layer_values(a) * learning_rate);
     end
end


%error term for hidden layer
hid_layer_error_term = [];
r = 1;
sum = 0;
 for b = 1:30
     for c = 1:10
     value = weight_vector(c) * w2(b,c); % Check this change
     sum = sum +value;
     end
      hid_layer_error_term(end+1) = sum;
      sum = 0;
 end
%weight update
 
row = 1;
for a = 1:784
     for b = 1:30
     p =  hidden_layer_values(b);
     if  p == 1
         hidden_layer_values(b) = 0.99;
     end
      if  p == 0
         hidden_layer_values(b) = 0.01;
     end
     weight_update1 = hid_layer_error_term(b) * hidden_layer_values(b) * (1-hidden_layer_values(b)) * train.X(a+1,q) * learning_rate;  
     w1(a,b) = w1(a,b) + weight_update1;
     end
     weight_update1 = 0;
end
end
disp('out from loop');
w1(1,:)

%now testing
count = 0;
for k = 1:10000
x1= [];
x2 = [];
v1= [];
v2 =[];
hidden_layer_values = [];
for b = 1:30
  v1 = transpose(w1(:,b));
  v2 = test.X(:,k);
  %whos
  v = v1 * v2;
  hidden_layer_values(end+1) =  1 ./ (1+ exp(-(v))); 
end
hidden_layer_values;
output_layer_values = [];

for b = 1:10
     x1 = transpose(w2(:,b));
     x2 = transpose(hidden_layer_values(1,:));
     %whos
     output_layer_values(end+1) =  x1 * x2;
end
output_layer_values;

softmax_values = [];
ans = 0;
sum_ans = 0;

%denominator factor
for c = 1:10 %number of columns
    sum_ans = sum_ans + exp(output_layer_values(c));
end

for b = 1:10 %number of columns
    ans = exp(output_layer_values(b));
    softmax_values(end +1) = ans / sum_ans;
    ans = 0;
end
softmax_values;
% computing error
error_values = [];
v = [];
h = [];
v = softmax_values - target_softmax;
y = v.^2; %squaring each value
h = y * 0.5;
[M,I] = min(h);
 if I == test.y(1,k)
 accuracy = accuracy + 1;
 
 end
count = k;
end

accuracy
percent_Accuracy =  (accuracy/count) * 100

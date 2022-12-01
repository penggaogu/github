%% 2nd Order Taylor Method

%% Inputs

a = 0;          % left endpoint
b = 1;          % right endpoint
h = 0.1;        % stepsize
N = (b-a)/h;    % the number of steps
alpha = 1;      % initial y value

f = @(t,y) (2-2*t*y)/(t^2+1);        % as in dy/dt = f(t,y);
df = @(t,y) (-8*t-2*y+6*t^2*y)/(1+t^2)^2;


%% Order 2

t = zeros(1,N+1);       % stores all the t values
w = zeros(1,N+1);       % stores all the approximation values for order 2

t(1) = a;
w(1) = alpha;

for i=1:N
    w(i+1) = w(i) + h*f(t(i),w(i)) + (h^2/2)*df(t(i),w(i));
    t(i+1) = a + i*h;
end

%% Compute the actual errors, error bound, and print information

error = zeros(1,N+1);
error_order_4 = zeros(1,N+1);
%error_bound = zeros(1,N+1);
%M = 1;
%L = 1;
fprintf('i\tt_i\t\tw_i\n')

for i=1:N+1           
    fprintf('%d\t%.9f\t%.9f\n',i,t(i),w(i))
end

















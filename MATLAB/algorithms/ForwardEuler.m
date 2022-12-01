%% Forward Euler

%% Inputs

a = 1;          % left endpoint
b = 2;          % right endpoint
h = 0.25;        % stepsize
N = (b-a)/h;    % the number of steps
alpha = 2;      % initial y value

f = @(t,y) 1 + y/t;        % as in dy/dt = f(t,y);

%% Forward Euler

t = zeros(1,N+1);       % stores all the t values
w = zeros(1,N+1);       % stores all the approximation values

t(1) = a;
w(1) = alpha;

for i=1:N
    w(i+1) = w(i) + h*f(t(i),w(i));
    t(i+1) = a + i*h;
end

%% Plot the approximation

figure()
plot(t,w,'*-')
hold on;            % so we can plot multiple things on the same graph

%% Plot the exact solution

y = @(t) t*log(t) + 2*t;

num_plot = 100;     % need a lot of plotting points to get a smooth graph!

t_plot = zeros(1,num_plot+1);
y_plot = zeros(1,num_plot+1);
h_plot = (b-a)/num_plot;

for i=1:num_plot+1
    t_plot(i) = a + (i-1) * h_plot;
    y_plot(i) = y(t_plot(i));
end

plot(t_plot,y_plot)
title("Forward Euler Method for solving y' = 1 + y/t, 1 \leq t \leq 2")
legend("Approximation","Exact Solution")

%% Compute the actual errors, error bound, and print information

error = zeros(1,N+1);
error_bound = zeros(1,N+1);
M = 1;
L = 1;
fprintf('i\tt_i\t\tw_i\t\ty(t_i)\t\t|y(t_i) - w_i)|\tbound\n')

for i=1:N+1
    error(i) = abs( y(t(i)) - w(i) );                 % | y(t_i) - w_i |
    error_bound(i) = (h*M/2*L) * (exp(L*(t(i)-a)) - 1 );
    fprintf('%d\t%.9f\t%.9f\t%.9f\t%.9f\t%.9f\n',i,t(i),w(i),y(t(i)),error(i),error_bound(i))
end

%% Plot the error

figure()
plot(t,error,'*-')
title("Error using Euler Method to solve y' = 1 + y/t, 1 \leq t \leq 2")
















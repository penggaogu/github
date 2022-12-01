%% Runge Kutta Order 4

%% Inputs
a = 1;          % left endpoint
b = 3;          % right endpoint
h = 0.2;        % stepsize
N = (b-a)/h;    % the number of steps
alpha = 0;      % initial y value
f = @(t,y) 1 + y/t + (y/t)^2;        % as in dy/dt = f(t,y);

%% Runge Kutta Order 4

t = zeros(1,N+1);       % stores all the t values
w = zeros(1,N+1);       % stores all the approximation values

t(1) = a;
w(1) = alpha;

for i=1:N
    t(i+1) = a + i*h;
    k1 = h * f(t(i),w(i));
    k2 = h*f( t(i) + h/2, w(i) + k1/2 );
    k3 = h*f( t(i) + h/2, w(i) + k2/2 );
    k4 = h*f( t(i+1), w(i) + k3 );
    w(i+1) = w(i) + (1/6)*(k1 + 2*k2 + 2*k3 + k4);
end

%% Plot the approximation

figure()
plot(t,w,'*-')
hold on;            % so we can plot multiple things on the same graph

%% Plot the exact solution

y = @(t) t*tan(log(t));

num_plot = 100;     % need a lot of plotting points to get a smooth graph!

t_plot = zeros(1,num_plot+1);
y_plot = zeros(1,num_plot+1);
h_plot = (b-a)/num_plot;

for i=1:num_plot+1
    t_plot(i) = a + (i-1) * h_plot;
    y_plot(i) = y(t_plot(i));
end

plot(t_plot,y_plot)
title("Runge Kutta Order 4 for solving y' = -(y+1)(y+3), 0 \leq t \leq 2")
legend("Approximation","Exact Solution")

%% Compute the actual errors, error bound, and print information

error = zeros(1,N+1);
fprintf('i\tt_i\t\tw_i\t\ty(t_i)\t\t|y(t_i) - w_i)|\n')

for i=1:N+1
    error(i) = abs( y(t(i)) - w(i) );                 % | y(t_i) - w_i |
    fprintf('%d\t%.9f\t%.9f\t%.9f\t%.9f\n',i-1,t(i),w(i),y(t(i)),error(i))
end

%% Plot the error

figure()
plot(t,error,'*-')
title("Error using Runge Kutta Order 4 to solve y' = -(y+1)(y+3), 0 \leq t \leq 2")
















%% Runge-Kutta Method for Systems of Differential Equations

%% Input information
a = 0;          % left endpoint
b = 2;          % right endpoint
m = 2;          % number of equations
h = 0.1;        % stepsize
N = (b-a)/h;    % number of subintervals
alpha1 = 0;  % initial conditions
alpha2 = -1;

f1 = @(t,u1,u2) -4*u1-2*u2+cos(t)+4*sin(t);
f2 = @(t,u1,u2) 3*u1+u2-3*sin(t);

% exact solutions
u1 = @(t) 2*exp(-t)-2*exp(-2*t)+sin(t);
u2 = @(t) -3*exp(-t)+2*exp(-2*t);

%% Do the method

t = a;

w1 = alpha1;
w2 = alpha2;
error_u1 = zeros(1,N+1);
error_u2 = zeros(1,N+1);
error_u1(1) = abs(w1-u1(t));
error_u2(1) = abs(w2-u2(t));

% output starting information
fprintf('t \t\t w1 \t\t u1 \t\t error_u1 \t w2 \t\t u2 \t\t error_u2 \n')             % header
fprintf('%f \t %f \t %f \t %f \t %f \t %f \t %f \n',t,w1,error_u1(1),0,w2,u2(t),error_u2(1));        % initial information

for i=1:N

    k(1,1) = h * f1(t, w1, w2);
    k(1,2) = h * f2(t, w1, w2);

    k(2,1) = h * f1(t + h/2, w1 + k(1,1)/2, w2 + k(1,2)/2);
    k(2,2) = h * f2(t + h/2, w1 + k(1,1)/2, w2 + k(1,2)/2);

    k(3,1) = h * f1(t + h/2, w1 + k(2,1)/2, w2 + k(2,2)/2);
    k(3,2) = h * f2(t + h/2, w1 + k(2,1)/2, w2 + k(2,2)/2);

    k(4,1) = h * f1(t + h, w1 + k(3,1), w2 + k(3,2));
    k(4,2) = h * f2(t + h, w1 + k(3,1), w2 + k(3,2));

    w1 = w1 + (k(1,1) + 2*k(2,1) + 2*k(3,1) + k(4,1))/6;
    w2 = w2 + (k(1,2) + 2*k(2,2) + 2*k(3,2) + k(4,2))/6;
    t = a + i*h;
    error_u1(i+1) = abs(w1-u1(t));
    error_u2(i+1) = abs(w2-u2(t));

    fprintf('%f \t %f \t %f \t %f \t %f \t %f \t %f \n',t,w1,u1(t),error_u1(i+1),w2,u2(t),error_u2(i+1));
end






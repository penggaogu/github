%% Linear Shooting

%% Input information

a = 1;      % left endpoint
b = 2;      % right endpoint
alpha = 1/2;  % boundary condition at left endpoint
beta = log(2);   % boundary condition at right endpoint
h = 0.1;     % number of subintervals

p = @(x) -4/x;
q = @(x) -2/x^2;
r = @(x) 2*log(x)/x^2;

%% Do the method

N = (b-a)/h;        % stepsize
u(1,1) = alpha;
u(2,1) = 0;
v(1,1) = 0;
v(2,1) = 1;
x = a;

fprintf('x \t\t u1 \t\t v2 \n')
fprintf('%f \t %f \t %f\n',x,u(1,1),v(1,1))

% there might be an error somewhere in this loop, u and v do not match the
% textbook example

for i=1:N
    x = a + (i-1)*h;

    k(1,1) = h * u(2,i);
    k(1,2) = h * ( p(x) * u(2,i) + q(x) * u(1,i) + r(x) );
    k(2,1) = h * ( u(2,i) + k(1,2)/2 );
    k(2,2) = h * ( p(x + h/2) * ( u(2,i) + k(1,2)/2 ) + q(x + h/2) * ( u(1,i) + k(1,1)/2 ) + r(x + h/2) );
    k(3,1) = h * ( u(2,i) + k(2,2)/2 );
    k(3,2) = h * ( p(x + h/2) * ( u(2,i) + k(2,2)/2 ) + q(x + h/2) * ( u(1,i) + k(2,1)/2 ) + r(x + h/2) );
    k(4,1) = h * ( u(2,i) + k(3,2) );
    k(4,2) = h * ( p(x + h) * ( u(2,i) + k(3,2) ) + q(x + h)*( u(1,i) + k(3,1) ) + r(x + h) );

    u(1,i+1) = u(1,i) + ( k(1,1) + 2*k(2,1) + 2*k(3,1) + k(4,1) )/6;
    u(2,i+1) = u(2,i) + ( k(1,2) + 2*k(2,2) + 2*k(3,2) + k(4,2) )/6;

    k_prime(1,1) = h * v(2,i);
    k_prime(1,2) = h * ( p(x) * v(2,i) + q(x) * v(1,i) );
    k_prime(2,1) = h * ( v(2,i) + k_prime(1,2)/2 );
    k_prime(2,2) = h * ( p(x + h/2) * ( v(2,i) + k_prime(1,2)/2 ) + q(x + h/2) * ( v(1,i) + k_prime(1,1)/2 ) );
    k_prime(3,1) = h * ( v(2,i) + k_prime(2,2)/2 );
    k_prime(3,2) = h * ( p(x + h/2) * ( v(2,i) + k_prime(2,2)/2 ) + q(x + h/2) * ( v(1,i) + k_prime(2,1)/2 ) );
    k_prime(4,1) = h * ( v(2,i) + k_prime(3,2) );
    k_prime(4,2) = h * ( p(x + h) * ( v(2,i) + k_prime(3,2) ) + q(x + h)*( v(1,i) + k_prime(3,1)) );

    v(1,i+1) = v(1,i) + (k_prime(1,1) + 2*k_prime(2,1) + 2*k_prime(3,1) + k_prime(4,1))/6;
    v(2,i+1) = v(2,i) + (k_prime(1,2) + 2*k_prime(2,2) + 2*k_prime(3,2) + k_prime(4,2))/6;

    fprintf('%f \t %f \t %f\n',x,u(1,i+1),v(1,i+1))
end

fprintf('\n')

w(1,1) = alpha;
w(2,1) = (beta - u(1,N+1))/(v(1,N+1));

fprintf('x \t\t w1 \t\t w2 \n')
fprintf('%f \t %f \t %f\n',a,w(1,1),w(2,1))

for i=1:N
    W1 = u(1,i+1) + w(2,1)*v(1,i+1);
    W2 = u(2,i+1) + w(2,1)*v(2,i+1);
    x = a + i*h;
    fprintf('%f \t %f \t %f\n',x,W1,W2)
end
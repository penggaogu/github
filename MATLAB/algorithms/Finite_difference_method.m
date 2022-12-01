%% Linear Finite-Difference Method
%% Input Information
A = 0;
B = 1;
alpha = 2;
beta = 1;
h = 0.1;
   % left endpoint
   % right endpoint
  % boundary condition at left endpoint
% boundary condition at right endpoint
p = @(x) -3;
q = @(x) 2;
r = @(x) 2*x + 3;
%% Do the method
N = 9;
x = A + h;
a = zeros(1, N+1);
b = zeros(1, N+1);
c = zeros(1, N+1);
d = zeros(1, N+1);
l = zeros(1, N+1);
u = zeros(1, N+1);
z = zeros(1, N+1);
a(1) = 2 + (h^2)*q(x);
b(1) = -1 + (h/2)*p(x);
d(1) = -(h^2)*r(x) + (1 + (h/2)*p(x))*alpha;
fprintf('x \t w_i \n')
for i=2:(N-1)
    x = A + i * h;
    a(i) = 2 + (h^2)*q(x);
    b(i) = -1 + (h/2)*p(x);
    c(i) = -1 - (h/2)*p(x);
    d(i) = -(h^2)*r(x);
end
x = B - h;
a(N) = 2 + (h^2)*q(x);
c(N) = -1 - (h/2)*p(x);
d(N) = -(h^2)*r(x) + (1 - (h/2)*p(x))*beta;
l(1) = a(1);
u(1) = b(1)/a(1);
z(1) = d(1)/l(1);
for i=2:(N-1)
    l(i) = a(i) - c(i)*u(i-1);
    u(i) = b(i)/l(i);
    z(i) = (d(i) - c(i)*z(i-1))/l(i);
end
l(N) = a(N) - c(N)*u(N-1);
z(N) = (d(N) - c(N)*z(N-1))/l(N);
w(1) = alpha;
w(N+1) = beta;
w(N) = z(N);
for i=N-1:1
    w(i) = z(i) - u(i)*w(i+1);
end
fprintf('%f \t %f \n',A,alpha)
for i=1:N+1
    x = A + i*h;
    fprintf('%f \t %f \n',x,w(i))
end
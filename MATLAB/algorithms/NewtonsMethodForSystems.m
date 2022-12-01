%% Newton's Method for Systems

%% Input Information

F = @(x) [log(x(1)^2+x(2)^2)-sin(x(1)*x(2))-log(2)-log(pi);
          exp(x(1)-x(2))+cos(x(1)*x(2))];

J = @(x) [(2*x(1))/(x(1)^2+x(2)^2)-x(2)*cos(x(1)*x(2)), (2*x(2))/(x(1)^2+x(2)^2)-x(1)*cos(x(1)*x(2));
          exp(x(1)-x(2))-x(2)*sin(x(1)*x(2)), -exp(x(1)-x(2))-x(1)*sin(x(1)*x(2))];

x = [2;
     2];

tol = 1e-6;                 % tolerance, 1e-4 = 10^{-4}

max_iter = 10;              % max number of iterations

%% Newton's Method for Systems

k = 1;                      % iteration count

fprintf('i\tx1\t\tx2\t\tinf error\n');          % for display
fprintf('%d\t%f\t%f\n',0,x(1),x(2));

while( k <= max_iter)

    % solve the system J(x)y = -F(x)
    y = J(x)\-F(x);

    x = x+y;
    
    % calculate infty norm
    inf_error = max(abs(y));

    % display information
    fprintf('%d\t%f\t%f\t%f\n',k,x(1),x(2),inf_error);    % displays iteration i, p_i
    
    % check stopping condition 
    if(inf_error < tol)
        break;
    end
    
    % increase iteration count
    k = k + 1;
    
end

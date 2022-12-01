%% Runge Kutta Fehlberg
%% Inputs
a = 1;              % left endpoint
b = 3;              % right endpoint
alpha = 0;        % initial y value
tol = 1e-6;         %tolerance
hmax = 0.5;        % maximum step size
hmin = 0.05;        % minimum step size
f = @(t,y) 1 + y/t + (y/t)^2;              % as in dy/dt = f(t,y);
y = @(t) t*tan(log(t));    % exact solution
%% Runge-Kutta-Fehlberg
t = a;
w = alpha;
h = hmax;
FLAG = 1;
N = (b-a)/hmin;
i = 1;
%j = 1;
fprintf('i \t t \t \t y_i = y(t_i) \t w_i \t\t h_i \t\t R \t\t abs(y(t)-w)\n')
fprintf('%d \t %f \t %f \t %f \n',0,t,y(t),w)
while(FLAG == 1 && i < N+1) %% j < N+1?
    format long
    K1 = h * f(t, w);
    K2 = h * f(t + h/4, w + K1/4);
    K3 = h * f(t + 3*h/8, w + 3*K1/32 + 9*K2/32);
    K4 = h * f(t + 12*h/13, w + 1932*K1/2197 - 7200*K2/2197 + 7296*K3/2197);
    K5 = h * f(t + h, w + 439*K1/216 - 8*K2 + 3680*K3/513 - 845*K4/4104);
    K6 = h * f(t + h/2, w - 8*K1/27 + 2*K2 - 3544*K3/2565 + 1859*K4/4104 - 11*K5/40);
    R = (1/h)*abs( K1/360 - 128*K3/4275 - 2197*K4/75240 + K5/50 + 2*K6/55 );    % approximates the LTE
    if(R <= tol)
        t = t + h;
        w = w + 25*K1/216 + 1408*K3/2565 + 2197*K4/4104 - K5/5;
        i = i+1;
        fprintf('%d \t %f \t %f \t %f \t %f \t %.9f \t %.9f \n',i-1,t,y(t),w,h,R,abs(y(t)-w))
        % move output stuff here?
        % else
        %     disp("Ooops, might need to adjust the indices....");
    end
    % choose a new stepsize 
    delta = 0.84*(tol/R)^(1/4);
    if(delta <= 0.1)
        h = 0.1*h;
    else
        if(delta >= 4)
            h = 4*h;
        else
            h = delta*h;
        end
    end

    % check step size
    if(h > hmax)
        h = hmax;
    end
    if(t >= b)
        FLAG = 0;
    else
        if(t + h > b)
            h = b - t;
        else
            if(h < hmin)
                FLAG = 0;
                disp("Minimum h exceeded");
            end
        end
    end
end










function [choice, RT] = LBA_Simulation(A, b, v, t0, sv,silent)

%   1. Input are now matrices of size n by N 
%         - N is the number of accumulators.
%         - n is the number of trials
% Usage: [choice RT conf] = LBA_trial(A, b, v, t0, sv, N)

% Outputs:
%
% choice = scalar from 1:N indicating response chosen by model
% RT = reaction time in ms
% confidence = confidence computed using balance of evidence rule 
% (Vickers, D. (1979). Decision Processes in Visual Perception.)
%
% SF 2012?

[n, N] = size(A); 
% n = number of simulated trials; N = number of choices (accumulators)

k = zeros(n,N);     d = zeros(n,N);     t = zeros(n,N);
RT = zeros(n,1);    choice = zeros(n,1);

for i = 1:n
    trialOK = false;
    if silent == false && mod(i,ceil(0.25*n)) == 0 
        disp([' Simulated data: ',num2str(round(100*i/n,0)),' % completed !'])
    end
    while ~trialOK
        for j = 1:N
            
            % Get starting point
            k(i,j) = rand().*A(i,j);
            
            % Get drift rate
            d(i,j) = normrnd(v(i,j), sv(i,j));
            
            % Get time to threshold
            t(i,j) = (b(i,j)-k(i,j))./d(i,j);
            
            % Add on non-decision time
            allRT(i,j) = t0(i,j) + t(i,j);
        end
        
        % Get choice and confidence
        [RT(i), choice(i)] = min(allRT(i,:));
        
        %         Confidence is equal to threshold minus value of next best accumulator at decision
        %         time
        
        c=1;
        for j = 1:N
            if j ~= choice(i)
                z(c) = t(i,choice(i)).*d(i,j) + k(i,j);
                c=c+1;
            end
        end
        [nb ~] = max(z);
        conf = b-nb;
        
        % Check we have not sampled negative drift(s)
        if RT(i) > 0 & nb > 0
            trialOK = true;
        end
    end
end
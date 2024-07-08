%%%% Modifiedd ADMM

function []=GF_GAD(A0,f0,GT)
%  Input: A: Adjacency matrix NxN
%         f: Graph signal matrix Nxp
%         GT: 0
%  Output:
%  Author: Meiby Ortiz-Bouza/Duc Vu
%  Address: Michigan State University, ECE
%  email: ortizbou@msu.edu/vuduc2@msu.edu

%%% Parameters
T=4;  
rho=1;
alpha=0.5;

[patient_num,n,p] = size(f0);


%hubnodes = zeros(64,20);
filter_all = zeros(56,n,n);
scores_all = zeros(56, n);
f_tilde_all = zeros(56,n,p);



names = [
            "FP1"
            "AF7"
            "AF3"
            "F1"
            "F3"
            "F5"
            "F7"
            "FT7"
            "FC5"
            "FC3"
            "FC1"
            "C1"
            "C3"
            "C5"
            "T7"
            "TP7"
            "CP5"
            "CP3"
            "CP1"
            "P1"
            "P3"
            "P5"
            "P7"
            "P9"
            "PO7"
            "PO3"
            "O1"
            "IZ"
            "OZ"
            "POZ"
            "PZ"
            "CPZ"
            "FPZ"
            "FP2"
            "AF8"
            "AF4"
            "AFZ"
            "FZ"
            "F2"
            "F4"
            "F6"
            "F8"
            "FT8"
            "FC6"
            "FC4"
            "FC2"
            "FCZ"
            "CZ"
            "C2"
            "C4"
            "C6"
            "T8"
            "TP8"
            "CP6"
            "CP4"
            "CP2"
            "P2"
            "P4"
            "P6"
            "P8"
            "P10"
            "PO8"
            "PO4"
            "O2"
        ];

for patient = 1:patient_num
A = squeeze(A0(patient,:,:));
f = squeeze(f0(patient,:,:));
%A = A0;
%f = f0;
[~,N]=size(A);
[~,p]=size(f);  


%% Learn filter
An=normadj(A);    % normalized Adjacency
Ln = eye(N)-An;   % normalized Laplacian
[U,d]=eig(full(Ln));
D=diag(d);

%%%  t-th shifted input signal as S(t) := U'*D^t*U'*F
for t=1:T
zt{t}=U*d^(t-1)*U'*f;
end

for i=1:N
    for t=1:T
    zn(t,:,i)=zt{t}(i,:);
    end
end

%% Initializations
mu1=rand(N,p);
V=mu1/rho;
h=rand(T,1);
h=h/norm(h);
H=0;
for t=1:T
    Hnew=H+h(t)*diag(D.^(t-1));
    H=Hnew;  
end

thr=alpha/rho;
for n=1:40
    %% ADMM (Z,h,V)
    %%% B^(k+1) update using h^k and V^k
    X=(eye(N)-U*H*U')*f-V;
    B=wthresh(X,'s',thr);
    %%% h^(k+1) update using B^(k+1) and V^k
    E=B-f+V;
    count1=0;
    count2=0;
    SZ=0;
    for t=1:p       
        %This is the equivalent of allsumofp(S(:,:,p)*L*S(:,:,p)')
        ZN1 = permute(zn(:,t,:), [1,3,2])*Ln*permute(zn(:,t,:), [1,3,2])';
        SZnew = SZ + ZN1;
        SZ = SZnew;
        for k=1:N
            count2=count2+1;
            ZN2(:,:,count2)=zn(:,t,k)*zn(:,t,k)';
            b(:,:,count2)=zn(:,t,k)*E(k,t);
        end
    end

    Y=2*SZ+rho*sum(ZN2,3);
    h_new=-inv(Y)*rho*sum(b,3);
    h_new=h_new/norm(h_new);

    H=0; %% C filter for next iteration
    for t=1:T
        Hnew=H+h_new(t)*diag(D.^(t-1));
        H=Hnew;  
    end

    %%% V^(k+1) update using V^k, Z^(k+1), and c^(k+1)
    V_new=V+rho*(B-(eye(N)-U*H*U')*f);
    if norm(h_new-h)<10^-3
        break
    end
    h=h_new;
    V=V_new;
end
clear b ZN2


f_tilde=U*H*U'*f;
%testing direct solve
f_pseudo_inverse = pinv(f);
H = U'*f_tilde*f_pseudo_inverse*U;
plot(D,abs(H));
grid on
hold on
f_tilde = (eye(N) + 0.5*Ln)^(1/2)*f;
%plot(D,diag(abs(H)));
%testing
% diag_H = diag(H);
% for i=1:N
%     diag_H(i) = 1;
%     if i > 120 %arbitrary number to get low pass
%         diag_H(i) = 0;
%     end
% end
% H = diag(diag_H);
% plot(D,diag(abs(H)));
% f_tilde = U*H*U'*f;
% H = diag(1 - diag(abs(Ln)));
% f_tilde=Ln*f;
P = U'*f;
Q = U'*f_tilde;
for i=1:N
    h(i) = Q(i, 5)/P(i,5);
end
H = diag(1 - h);
%plot(D,diag(abs(H)));
% f_Tilde = U*H*U'*f;
f_tilde_all(patient,:,:) = f_tilde;
filter_all(patient,:,:) = H;
% 
win_length = 22;
[rows, cols] = size(f);
%%% Anomaly scoring with lasso
% for i=1:N
%     norm_vec = zeros(1, 360);
%     norm_vec_tilde = zeros(1, 360);
%     local_grad = (f - f(i,:))';
%     local_grad_tilde = (f_tilde - f_tilde(i,:))';
%     for start_col = 1:(cols -  win_length + 1)
%         window = local_grad(start_col:(start_col + win_length - 1), :);
%         window_tilde = local_grad_tilde(start_col:(start_col + win_length - 1), :);
%         norm_vec= norm_vec + vecnorm(window);
%         norm_vec_tilde = norm_vec_tilde + vecnorm(window_tilde);
%     end
%     s=A(i,:).*norm_vec;
%     e0(i)=sum(s);
% 
%     s=A(i,:).*norm_vec_tilde;
%     en(i)=sum(s);
% end
%% Anomaly scoring based on smoothness
scores=vecnorm(f'-f_tilde');

for i=1:N
    s=A(i,:).*vecnorm((f-f(i,:))');
    e0(i)=sum(s);
end

for i=1:N
    s=A(i,:).*vecnorm((f_tilde-f_tilde(i,:))');
    en(i)=sum(s);
end

scores=e0-en;

% disp(scores);
%This is for normalizing degree
% normalize_degree = sum(A,2)';
% scores = scores./normalize_degree;

clear e0
clear en
clear s
scores = zscore(scores, 1, 'all');
threshold = 3;
pred = find(scores > threshold| scores < -threshold);
disp(patient);
%disp(names(pred));
scores_all(patient,pred) = 1;
hubnodes(pred,patient) = 1;


end
save("20240514_grafhubconnector_Duc_allfilteredsignalt4_v02.mat", 'f_tilde_all');
save("20240514_grafhubconnector_Duc_resultst4_v02.mat", 'hubnodes');
save("20240514_grafhubconnector_Duc_allfiltermatrixt4_v02.mat", 'filter_all');
save("20240514_grafhubconnector_Duc_allhubscorest4_v02.mat", 'scores_all');
end

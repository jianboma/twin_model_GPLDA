function scores = score_twin_model_gplda_trials(plda, model_iv, test_iv,Kmodel,Ktest)
% computes the verification scores as the log-likelihood ratio of the same 
% versus different speaker models hypotheses in twin model GPLDA.
%
% Inputs:
%   plda            : structure containing twin model GPLDA hyperparameters
%   model_iv        : data matrix for enrollment i-vectors (column observations)
%   test_iv         : data matrix for test i-vectors (one observation per column)
%
% Outputs:
%    scores         : output verification scores matrix (all model-test combinations)
%
% References:
%   [1] D. Garcia-Romero and C.Y. Espy-Wilson, "Analysis of i-vector length 
%       normalization in speaker recognition systems," in Proc. INTERSPEECH,
%       Florence, Italy, Aug. 2011, pp. 249-252.
%   [2] J. Ma, V. Sethu, E. Ambikairajah, and K. A. Lee, "Twin Model G-PLDA
%       for Duration Mismatch Compensation in Text-Independent Speaker 
%       Verification," Interspeech 2016, pp. 1853-1857, 2016.
%
%
% Jianbo Ma <jianbo.ma@student.unsw.edu.au>
% EE&T, UNSW, Sydney

if ~isstruct(plda),
	fprintf(1, 'Error: plda should be a structure!\n');
	return;
end
ndim = size(model_iv,1);
Phi_short     = plda.Phi(1:ndim,:);
Phi_long     = plda.Phi(ndim+1:end,:);

Sigma_short   = plda.Sigma(1:ndim,1:ndim);
Sigma_long   = plda.Sigma(ndim+1:end,ndim+1:end);


W_short       = plda.W_short;
M_short       = plda.M_short;
W_long       = plda.W_long;
M_long       = plda.M_long;


%%%%% post-processing the model i-vectors %%%%%
model_iv = bsxfun(@minus, model_iv, M_long); % centering the data
model_iv = length_norm(model_iv); % normalizing the length
model_iv = W_long' * model_iv; % whitening data

%%%%% post-processing the test i-vectors %%%%%
test_iv = bsxfun(@minus, test_iv, M_short); % centering the data
test_iv = length_norm(test_iv); % normalizing the length
test_iv  = W_short' * test_iv; % whitening data

%%conventional way
ZERO_BLOCK = zeros(size(Phi_short,1),size(Phi_short,1));
A = Phi_long*Phi_long'+Sigma_long;
B = Phi_long*Phi_short';
C = Phi_short*Phi_long';
D = Phi_short*Phi_short'+Sigma_short;
Sigma_same = [A B;
              C D];
Sigma_same_inv = inv(Sigma_same);
Sigma_diff = [A ZERO_BLOCK;
              ZERO_BLOCK D];
Sigma_diff_inv = inv(Sigma_diff);

scores = zeros(size(Kmodel,1),1);
for num = 1:size(Kmodel)
    model_test_vector = [model_iv(:,Kmodel(num));test_iv(:,Ktest(num))];
    score1 = model_test_vector'*Sigma_same_inv*model_test_vector;
    score2 = model_test_vector'*Sigma_diff_inv*model_test_vector;
    scores(num) = 0.5*(score2-score1);
end
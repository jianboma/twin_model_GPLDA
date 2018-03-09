function plda = Twin_model_gplda_em(data_long, spk_labs_long,data_short,spk_labs_short, nphi, niter)
% trains a twin model GPLDA model 
% Inputs:
%   data_long            : input data matrix of long utterances, one observation per column
%   data_short            : input data matrix of short utterances, one observation per column
%   spk_labs_long        : class labels for observations in data matrix of long utterances
%   spk_labs_short        : class labels for observations in data matrix of short utterances
%   nphi            : dimensionality of the Eigenvoice subspace 
%   niter           : number of EM iterations for learning PLDA model
%
% Outputs:
%    plda           : a structure containing the twin model PLDA hyperparameters
%					  as well as the mean of development data and a whitening 
%					  transform:(plda.Phi_long: Eigenvoice matrix for long utteances, plda.Sigma_long: covariance
%					  matrix of the residual noise for long utteances, plda.Phi_short: Eigenvoice matrix for short utteances, plda.Sigma_short: covariance
%					  matrix of the residual noise for short utteances; plda.M: mean, plda.W: whitening transform)
%
% References:
%   [1] S.J.D. Prince and J.H. Elder, "Probabilistic linear discriminant analysis
%       for inferences about identity," in Proc. IEEE ICCV, Rio de Janeiro, Brazil,
%       Oct. 2007.
%   [2] D. Garcia-Romero and C.Y. Espy-Wilson, "Analysis of i-vector length 
%       normalization in speaker recognition systems," in Proc. INTERSPEECH,
%       Florence, Italy, Aug. 2011, pp. 249-252.
%   [3] P. Kenny, "Bayesian speaker verification with heavy-tailed priors," 
%       in Proc. Odyssey, The Speaker and Language Recognition Workshop, Brno, 
%       Czech Republic, Jun. 2010.
%   [4] J. Ma, V. Sethu, E. Ambikairajah, and K. A. Lee, "Twin Model G-PLDA
%       for Duration Mismatch Compensation in Text-Independent Speaker 
%       Verification," Interspeech 2016, pp. 1853-1857, 2016.
%
%
% Jianbo Ma <jianbo.ma@student.unsw.edu.au>
% EE&T, UNSW, Sydney

data = [data_short data_long];
spk_labs = [spk_labs_long;spk_labs_short];

[ndim, nobs] = size(data);
% clear data;
if ( nobs ~= length(spk_labs) ),
	error('oh dear! number of data samples should match the number of labels!');
end
clear spk_labs;

[spk_labs_long, I] = sort(spk_labs_long);
data_long = data_long(:, I);
[~, ia, ic] = unique(spk_labs_long, 'stable');
spk_counts_long = histc(ic, 1 : numel(ia)); % # sessions per speaker

[spk_labs_short, I] = sort(spk_labs_short);
data_short = data_short(:, I);
[~, ia, ic] = unique(spk_labs_short, 'stable');
spk_counts_short = histc(ic, 1 : numel(ia)); % # sessions per speaker

%---------------------------whitening data-------------------------%
M = mean(data, 2);
W1   = calc_white_mat(cov(data'));

M_short = M;
data_short = bsxfun(@minus, data_short, M_short); % centering the data
data_short = length_norm(data_short); % normalizing the length

W1_short = W1;
data_short = W1_short'*data_short;

M_long = M;
data_long = bsxfun(@minus, data_long, M_long); % centering the data
data_long = length_norm(data_long); % normalizing the length

W1_long = W1;
data_long = W1_long'*data_long;

%---------------------------form data-------------------------------------%
spk_counts = spk_counts_short;
nspks    = size(spk_counts, 1);
data = [];
spk_labs_short_uniq = unique(spk_labs_short);

for spk = 1 : nspks
    % Speaker indices
  
    whshort = ismember(spk_labs_short,spk_labs_short_uniq{spk});
    idx_short = find(whshort);
    session_short = length(idx_short);  
    Data_short = data_short(:, idx_short);
    
    whlong = ismember(spk_labs_long,spk_labs_short_uniq{spk});
    idx_long = find(whlong);
    session_long = length(idx_long);  
    Data_long = data_long(:, idx_long);
    
    %data re-arrange
%     num_re_order_randm = randi([1 session_short],session_short,1);
    num_re_order_randm = randi([1 session_long],session_short,1);
    data_long_duplication = Data_long(:,num_re_order_randm);
    data_augment = [Data_short;data_long_duplication];
end
%---------------------------initialize parameters-------------------------%
fprintf('\n\nRandomly initializing the PLDA hyperparameters ...\n\n');
% Initialize the parameters randomly
%----initialize for long utterance---%
[s1, s2] = RandStream.create('mrg32k3a', 'NumStreams', 2);
Sigma_long    = 100 * randn(s1, ndim); % covariance matrix of the residual term
Phi_long = randn(s2, ndim, nphi); % factor loading matrix (Eignevoice matrix)
Phi_long = bsxfun(@minus, Phi_long, mean(Phi_long, 2));
W2_long   = calc_white_mat(Phi_long' * Phi_long);
Phi_long = Phi_long * W2_long; % orthogonalize Eigenvoices (columns)
%----initialize for short utterance---%
[s1, s2] = RandStream.create('mrg32k3a', 'NumStreams', 2);
Sigma_short    = 100 * randn(s1, ndim); % covariance matrix of the residual term
Phi_short = randn(s2, ndim, nphi); % factor loading matrix (Eignevoice matrix)
Phi_short = bsxfun(@minus, Phi_short, mean(Phi_short, 2));
W2_short   = calc_white_mat(Phi_short' * Phi_short);
Phi_short = Phi_short * W2_short; % orthogonalize Eigenvoices (columns)
%------------------combine-------------------%
ZERO_BLOCK = zeros(ndim,ndim);
Sigma = [Sigma_short ZERO_BLOCK;
         ZERO_BLOCK Sigma_long];
Phi = [Phi_short;Phi_long];

%-------------------------start to train----------------------%

fprintf('Re-estimating the Eigenvoice subspace with %d factors ...\n', nphi);
for iter = 1 : niter
    fprintf('EM iter#: %d \t', iter);
    tim = tic;
    [Ey_short,Ey_long, Eyy_short,Eyy_long] = expectation_twin_model_plda(data,data_long,data_short, Phi, Sigma, spk_counts_long,spk_counts_short);
    [Phi, Sigma] = maximization_twin_model_plda(data_long,data_short, Ey_short,Ey_long, Eyy_short,Eyy_long);
    tim = toc(tim);
    fprintf('[elaps = %.2f s]\n', tim);
end

plda.Phi   = Phi;
plda.Sigma = Sigma;
plda.W_short     = W1_short;
plda.W_long     = W1_long;
plda.M_short    = M_short;
plda.M_long    = M_long;

function [Ey_short,Ey_long, Eyy_short,Eyy_long] = expectation_twin_model_plda(data,data_long,data_short, Phi, Sigma, spk_counts_long,spk_counts_short)

%---------------------re-arragne data, augment short and long data-----%

%----------------------------------------------------------%


% computes the posterior mean and covariance of the factors
spk_counts = spk_counts_short;
nphi     = size(Phi, 2);
nsamples_short = size(data_short, 2);
nsamples_long = size(data_long, 2);
nspks    = size(spk_counts, 1);

Ey_short = zeros(nphi, nsamples_short);
Ey_long = zeros(nphi, nsamples_long);
Eyy_short = zeros(nphi);
Eyy_long = zeros(nphi);
% initialize common terms to save computations
uniqFreqs  	  = unique(spk_counts);
nuniq 		  = size(uniqFreqs, 1);
invTerms      = cell(nuniq, 1);
invTerms(:)   = {zeros(nphi)};
PhiT_invS_Phi = ( Phi'/Sigma ) * Phi;
I = eye(nphi);
for ix = 1 : nuniq
    nPhiT_invS_Phi = uniqFreqs(ix) * PhiT_invS_Phi;
    Cyy =  pinv(I + nPhiT_invS_Phi);
    invTerms{ix} = Cyy;
end

data = Sigma\data;
% inv_Sigma = inv(Sigma);
cnt  = 1;
cnt_long = 1;
for spk = 1 : nspks
    nsessions_short = spk_counts_short(spk);
    nsessions_long = spk_counts_long(spk);
    % Speaker indices
    idx_long = cnt_long : ( cnt_long - 1 ) + spk_counts_long(spk);
    cnt_long  = cnt_long + spk_counts_long(spk);
    
    idx = cnt : ( cnt - 1 ) + spk_counts(spk);
    cnt  = cnt + spk_counts(spk);
    Data = data(:, idx);
    PhiT_invS_y = sum(Phi' * Data, 2);
    Cyy = invTerms{ uniqFreqs == nsessions_short };
    Ey_spk  = Cyy * PhiT_invS_y;
    Eyy_spk = Cyy + Ey_spk * Ey_spk';
    Eyy_short     = Eyy_short + nsessions_short * Eyy_spk;
    Ey_short(:, idx) = repmat(Ey_spk, 1, nsessions_short);
    
    Eyy_long     = Eyy_long + nsessions_long * Eyy_spk;
    Ey_long(:, idx_long) = repmat(Ey_spk, 1, nsessions_long);
end

function [Phi, Sigma] = maximization_twin_model_plda(data_long,data_short, Ey_short,Ey_long, Eyy_short,Eyy_long)
% ML re-estimation of the Eignevoice subspace and the covariance of the
% residual noise (full).

nsamples_short = size(data_short, 2);
Data_sqr_short = data_short * data_short';
Phi_short      = data_short * Ey_short' / (Eyy_short);
Sigma_short    = 1/nsamples_short * (Data_sqr_short - (Phi_short * Ey_short) * data_short');

nsamples_long = size(data_long, 2);
Data_sqr_long = data_long * data_long';
Phi_long      = data_long * Ey_long' / (Eyy_long);
Sigma_long    = 1/nsamples_long * (Data_sqr_long - (Phi_long * Ey_long) * data_long');


ndim = size(data_long, 1);
ZERO_BLOCK = zeros(ndim,ndim);
Sigma = [Sigma_short ZERO_BLOCK;
         ZERO_BLOCK Sigma_long];
Phi = [Phi_short;Phi_long];

function W = calc_white_mat(X)
% calculates the whitening transformation for cov matrix X
[~, D, V] = svd(X);
W = V * diag(sparse(1./( sqrt(diag(D)) + 1e-10 )));

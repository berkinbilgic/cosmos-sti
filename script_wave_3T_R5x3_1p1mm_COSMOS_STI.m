%% load data

spatial_res = [255/240, 255/240, 180/168];      % spatial resolution

load phase_tissue_12orient                      % registered tissue phase at 12 head orientations
load msk                                        % brain mask
load Rotation_Matrices                          % FSL FLIRT rotation matrices

mosaic(-imrotate(mean(phase_tissue(15:end-15,:,60:5:end,:),4), -90), 2, 5, 1, 'Average Tissue Phase', [-.045,.045])


%% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%% +++++++++++++++++++++++++++++ COSMOS RECON +++++++++++++++++++++++++++++ 
%% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%% create susceptibility kernels for each orientation

N = size(msk);
center = N/2 + 1;

[ky,kx,kz] = meshgrid(-N(1)/2:N(1)/2-1, -N(2)/2:N(2)/2-1, -N(3)/2:N(3)/2-1);

kx = (kx / max(abs(kx(:)))) / spatial_res(1);
ky = (ky / max(abs(ky(:)))) / spatial_res(2);
kz = (kz / max(abs(kz(:)))) / spatial_res(3);

k2 = kx.^2 + ky.^2 + kz.^2;

kernel = zeros(size(phase_tissue));

for t = 1:size(R_tot,3)
    kernel(:,:,:,t) = fftshift( 1/3 - (kx * R_tot(3,1,t) + ky * R_tot(3,2,t) + kz * R_tot(3,3,t)).^2 ./ (k2 + eps) );    

    mosaic(fftshift(kernel(:,:,:,t)), 8, 20, 2, ['Orientation: ', num2str(t)], [-2/3,1/3])
end

kernel_sum = fftshift(mean(abs(kernel), 4));
mosaic(squeeze(kernel_sum(1+end/2,:,:)), 1, 1, 3, 'Average Kernel', [0,2/3])


%% Thresholded K-space Division (TKD) reconstruction for each orientation

kthre = 0.1;        % TKD truncation threshold

chi_tkd = zeros(size(phase_tissue));
for t = 1:size(phase_tissue,4)
    krnl = kernel(:,:,:,t);
    kernel_inv = zeros(N);
    kernel_inv( abs(krnl) > kthre ) = 1 ./ krnl(abs(krnl) > kthre);

    chi_tkd(:,:,:,t) = real( ifftn( fftn(phase_tissue(:,:,:,t)) .* kernel_inv ) ) .* msk; 

    mosaic(imrotate(mean(chi_tkd(15:end-15,:,60:5:end,1:t),4), -90), 2, 5, 4, 'Average TKD', [-.14,.14]), 
end

    
%% COSMOS reconstruction with closed-form solution

Phase_tissue = zeros(size(phase_tissue));

for t = 1:size(phase_tissue,4)
    Phase_tissue(:,:,:,t) = fftn(phase_tissue(:,:,:,t));
end

kernel_sum = sum(abs(kernel).^2, 4);

chi_cosmos = real( ifftn( sum(kernel .* Phase_tissue, 4) ./ (eps + kernel_sum) ) ) .* msk;

mosaic(imrotate(chi_cosmos(15:end-15,:,60:5:end), -90), 2, 5, 5, 'COSMOS', [-.14,.14]), 


%% display in 3d

ay_pos = N(1)/2; 
cey_pos = N(2)/2;
zi_pos = 76;

scale_fig = [-.037, .037];
avg_size = 1;

fig_num = 1;
img_disp = -mean(phase_tissue,4);
img_disp(img_disp==0) = -1;

figure(fig_num), subplot(131), imagesc(flipud(squeeze(mean(img_disp(ay_pos-avg_size:ay_pos+avg_size,:,:),1)).'), scale_fig), axis image off, colormap gray
figure(fig_num), subplot(132), imagesc(flipud((squeeze(mean(img_disp(:,cey_pos-avg_size:cey_pos+avg_size,:),2))).'), scale_fig), axis image off, colormap gray
title('Average Phase', 'fontsize', 24, 'color', [1,1,1])
figure(fig_num), subplot(133), imagesc(((mean(img_disp(:,:,zi_pos-avg_size:zi_pos+avg_size),3))).', scale_fig), axis image off, colormap gray, set(gcf, 'color', 'k')


scale_fig = [-.12, .14];

fig_num = 2;
img_disp = mean(chi_tkd, 4);
img_disp(img_disp==0) = -1;

figure(fig_num), subplot(131), imagesc(flipud(squeeze(mean(img_disp(ay_pos-avg_size:ay_pos+avg_size,:,:),1)).'), scale_fig), axis image off, colormap gray
figure(fig_num), subplot(132), imagesc(flipud((squeeze(mean(img_disp(:,cey_pos-avg_size:cey_pos+avg_size,:),2))).'), scale_fig), axis image off, colormap gray
title('Average TKD QSM', 'fontsize', 24, 'color', [1,1,1])
figure(fig_num), subplot(133), imagesc(((mean(img_disp(:,:,zi_pos-avg_size:zi_pos+avg_size),3))).', scale_fig), axis image off, colormap gray, set(gcf, 'color', 'k')


fig_num = 3;
img_disp = chi_cosmos;
img_disp(img_disp==0) = -1;

figure(fig_num), subplot(131), imagesc(flipud(squeeze(mean(img_disp(ay_pos-avg_size:ay_pos+avg_size,:,:),1)).'), scale_fig), axis image off, colormap gray
figure(fig_num), subplot(132), imagesc(flipud((squeeze(mean(img_disp(:,cey_pos-avg_size:cey_pos+avg_size,:),2))).'), scale_fig), axis image off, colormap gray
title('Cosmos QSM', 'fontsize', 24, 'color', [1,1,1])
figure(fig_num), subplot(133), imagesc(((mean(img_disp(:,:,zi_pos-avg_size:zi_pos+avg_size),3))).', scale_fig), axis image off, colormap gray, set(gcf, 'color', 'k')


%% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%% ++++++++++++++++++++++++++++++ STI RECON +++++++++++++++++++++++++++++++ 
%% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%% create direction vectors for each orientation

H_Matrix = zeros(size(R_tot,3), 3);
H_Deg = zeros(size(R_tot,3), 1);

for ndir = 1:size(R_tot,3)
    H_Vec = R_tot(1:3,1:3,ndir)' * [0;0;1];
    
    H_Matrix(ndir,:) = H_Vec;
    H_Deg(ndir) = acosd(H_Vec(3));
    
    figure(20), hold on, plot3(linspace(0, H_Vec(1), 100), linspace(0, H_Vec(2), 100), linspace(0, H_Vec(3), 100), 'color', rand(3,1))
    grid on, title('Direction vectors')
end
figure(21), plot(H_Deg, 'marker', '*'), axis tight, title('Degrees of rotation')


%% Iterative STI recon with LSQR

[ky,kx,kz] = meshgrid(-N(2)/2:N(2)/2-1,-N(1)/2:N(1)/2-1,-N(3)/2:N(3)/2-1);      % k-space grid
kx = fftshift(kx);      ky = fftshift(ky);      kz = fftshift(kz);

N_direction = size(phase_tissue, 4);        % no of directions

Phase_tissue = zeros(size(phase_tissue));
for ind = 1:N_direction
    Phase_tissue(:,:,:,ind) = fftn(phase_tissue(:,:,:,ind));
end

param = [];
param.SS = N;
param.N_direction = N_direction;
param.H_Matrix = H_Matrix;
param.kx = kx;      param.ky = ky;      param.kz = kz;
param.k2 = kx.^2 + ky.^2 + kz.^2;

lsqr_tol = 1e-2;                            % LSQR tolerance
lsqr_iter = 30;                             % no of LSQR iterations    

tic
    [res, flag, relres, iter] = lsqr(@apply_STI, Phase_tissue(:), lsqr_tol, lsqr_iter, [], [], [], param);  
    disp(['Flag: ', num2str(flag), '  Relres: ', num2str(relres), '  Iter: ', num2str(iter)])
toc

Fchi_res = reshape(res, [N,6]);             % susceptibility tensor in k-space

chi_res = zeros(size(Fchi_res));
for ind = 1:6
    chi_res(:,:,:,ind) = real( ifftn(Fchi_res(:,:,:,ind)) ) .* msk;             % susceptibility tensor in image space
end


%% display tensor components

ay_pos = 72; 
cey_pos = 80;
zi_pos = 70;

scale_fig = [-.11, .11];
avg_size = 1;

for t = 1:6
    img_disp = chi_res(:,:,:,t);
    
    img_disp(img_disp==0) = -1;
    
    fig_num = t + 10;
    figure(fig_num), subplot(131), imagesc(flipud(squeeze(mean(img_disp(ay_pos-avg_size:ay_pos+avg_size,:,:),1)).'), scale_fig), axis image off, colormap gray
    figure(fig_num), subplot(132), imagesc(flipud((squeeze(mean(img_disp(:,cey_pos-avg_size:cey_pos+avg_size,:),2))).'), scale_fig), axis image off, colormap gray
    title(['Tensor component: ' num2str(t), ' / 6'], 'fontsize', 24, 'color', [1,1,1])
    figure(fig_num), subplot(133), imagesc(((mean(img_disp(:,:,zi_pos-avg_size:zi_pos+avg_size),3))).', scale_fig), axis image off, colormap gray
    set(gcf, 'color', [0 0 0])
end


%% Compute eigenvalues at each voxel

Chi_tensor = zeros([N, 3, 3]);
Chi_tensor(:,:,:,1,1) = chi_res(:,:,:,1);
Chi_tensor(:,:,:,1,2) = chi_res(:,:,:,2);
Chi_tensor(:,:,:,2,1) = chi_res(:,:,:,2);
Chi_tensor(:,:,:,1,3) = chi_res(:,:,:,3);
Chi_tensor(:,:,:,3,1) = chi_res(:,:,:,3);
Chi_tensor(:,:,:,2,2) = chi_res(:,:,:,4);
Chi_tensor(:,:,:,2,3) = chi_res(:,:,:,5);
Chi_tensor(:,:,:,3,2) = chi_res(:,:,:,5);
Chi_tensor(:,:,:,3,3) = chi_res(:,:,:,6);

mask_tensor = msk(:);

Chi_tensor = permute(Chi_tensor, [4,5,1,2,3]);
chi_tensor = reshape(Chi_tensor, [3,3, numel(mask_tensor)]);

Chi_eig = zeros(numel(mask_tensor),3);

tic
for v = 1:length(mask_tensor)
    if mask_tensor(v) ~= 0
        [V,D] = eig(chi_tensor(:,:,v));
        Chi_eig(v,:) = diag(D)';
    end
end
toc

Chi_eig = reshape(Chi_eig, [N, 3]);


%% display eigenvalues

fig_num = 1;
scale_fig = [-.17, .17];

img_disp = Chi_eig(:,:,:,3);        % Lambda1 : most paramagnetic
img_disp(img_disp==0) = -1;

figure(fig_num), subplot(131), imagesc(flipud(squeeze(mean(img_disp(ay_pos-avg_size:ay_pos+avg_size,:,:),1)).'), scale_fig), axis image off, colormap gray
figure(fig_num), subplot(132), imagesc(flipud((squeeze(mean(img_disp(:,cey_pos-avg_size:cey_pos+avg_size,:),2))).'), scale_fig), axis image off, colormap gray
title('Lambda 1', 'color', 'w', 'fontsize', 24)
figure(fig_num), subplot(133), imagesc(((mean(img_disp(:,:,zi_pos-avg_size:zi_pos+avg_size),3))).', scale_fig), axis image off, colormap gray, set(gcf, 'color', [0 0 0])


fig_num = 2;

img_disp = Chi_eig(:,:,:,2);
img_disp(img_disp==0) = -1;

figure(fig_num), subplot(131), imagesc(flipud(squeeze(mean(img_disp(ay_pos-avg_size:ay_pos+avg_size,:,:),1)).'), scale_fig), axis image off, colormap gray
figure(fig_num), subplot(132), imagesc(flipud((squeeze(mean(img_disp(:,cey_pos-avg_size:cey_pos+avg_size,:),2))).'), scale_fig), axis image off, colormap gray
title('Lambda 2', 'color', 'w', 'fontsize', 24)
figure(fig_num), subplot(133), imagesc(((mean(img_disp(:,:,zi_pos-avg_size:zi_pos+avg_size),3))).', scale_fig), axis image off, colormap gray, set(gcf, 'color', [0 0 0])


fig_num = 3;

img_disp = Chi_eig(:,:,:,1);
img_disp(img_disp==0) = -1;

figure(fig_num), subplot(131), imagesc(flipud(squeeze(mean(img_disp(ay_pos-avg_size:ay_pos+avg_size,:,:),1)).'), scale_fig), axis image off, colormap gray
figure(fig_num), subplot(132), imagesc(flipud((squeeze(mean(img_disp(:,cey_pos-avg_size:cey_pos+avg_size,:),2))).'), scale_fig), axis image off, colormap gray
title('Lambda 3', 'color', 'w', 'fontsize', 24)
figure(fig_num), subplot(133), imagesc(((mean(img_disp(:,:,zi_pos-avg_size:zi_pos+avg_size),3))).', scale_fig), axis image off, colormap gray, set(gcf, 'color', [0 0 0])


%% display Mean Magnetic Susceptibility (MMS) and Magnetic Susceptibility Anisotropy (MSA)

MMS = mean(Chi_eig,4);
MSA = Chi_eig(:,:,:,3) - (Chi_eig(:,:,:,1) + Chi_eig(:,:,:,2)) / 2; 

fig_num = 4;
scale_fig = [-.073, .10];

img_disp = MMS;
img_disp(img_disp==0) = -1;

figure(fig_num), subplot(131), imagesc(flipud(squeeze(mean(img_disp(ay_pos-avg_size:ay_pos+avg_size,:,:),1)).'), scale_fig), axis image off, colormap gray
figure(fig_num), subplot(132), imagesc(flipud((squeeze(mean(img_disp(:,cey_pos-avg_size:cey_pos+avg_size,:),2))).'), scale_fig), axis image off, colormap gray
title('MMS', 'color', 'w', 'fontsize', 24)
figure(fig_num), subplot(133), imagesc(((mean(img_disp(:,:,zi_pos-avg_size:zi_pos+avg_size),3))).', scale_fig), axis image off, colormap gray, set(gcf, 'color', [0 0 0])


fig_num = 5;
scale_fig = [0, .17];

img_disp = MSA;
img_disp(img_disp==0) = -1;

figure(fig_num), subplot(131), imagesc(flipud(squeeze(mean(img_disp(ay_pos-avg_size:ay_pos+avg_size,:,:),1)).'), scale_fig), axis image off, colormap gray
figure(fig_num), subplot(132), imagesc(flipud((squeeze(mean(img_disp(:,cey_pos-avg_size:cey_pos+avg_size,:),2))).'), scale_fig), axis image off, colormap gray
title('MSA', 'color', 'w', 'fontsize', 24)
figure(fig_num), subplot(133), imagesc(((mean(img_disp(:,:,zi_pos-avg_size:zi_pos+avg_size),3))).', scale_fig), axis image off, colormap gray, set(gcf, 'color', [0 0 0])



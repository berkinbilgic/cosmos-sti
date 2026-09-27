function [ res, tflag ] = apply_STI( in, params, tflag )
%APPLY_COLLAPSE Summary of this function goes here
%   Detailed explanation goes here

if strcmp(tflag,'transp');
    im = reshape(in, [params.SS, params.N_direction]);
    
    Res = zeros([params.SS, 6]);

    for n = 1:params.N_direction
        H_Vec = params.H_Matrix(n,:);
        kH_over_k2 = (H_Vec(1) * params.kx + H_Vec(2) * params.ky + H_Vec(3) * params.kz) ./ (eps + params.k2);
        
        Res(:,:,:,1) = Res(:,:,:,1) + ((H_Vec(1)^2)/3 - H_Vec(1)*params.kx .* kH_over_k2) .* im(:,:,:,n);
        
        Res(:,:,:,2) = Res(:,:,:,2) + (2*(H_Vec(1)*H_Vec(2))/3 - (H_Vec(1)*params.ky + H_Vec(2)*params.kx) .* kH_over_k2) .* im(:,:,:,n);
        
        Res(:,:,:,3) = Res(:,:,:,3) + (2*(H_Vec(1)*H_Vec(3))/3 - (H_Vec(1)*params.kz + H_Vec(3)*params.kx) .* kH_over_k2) .* im(:,:,:,n);
        
        Res(:,:,:,4) = Res(:,:,:,4) + ((H_Vec(2)^2)/3 - H_Vec(2)*params.ky .* kH_over_k2) .* im(:,:,:,n);
        
        Res(:,:,:,5) = Res(:,:,:,5) + (2*(H_Vec(2)*H_Vec(3))/3 - (H_Vec(2)*params.kz + H_Vec(3)*params.ky) .* kH_over_k2) .* im(:,:,:,n);
        
        Res(:,:,:,6) = Res(:,:,:,6) + ((H_Vec(3)^2)/3 - H_Vec(3)*params.kz .* kH_over_k2) .* im(:,:,:,n);
    end
    
    res = Res(:);
    
else
    
    % 1/3 * (Fx11 * Hix^2 + Fx22 * Hiy^2 + Fx33 * Hiz^2 + 2* Fx12 * Hix*Hiy + 2*Fx13 * Hix*Hiz + 2*Fx23 * Hiy * Hiz)
    % - (Kx * Hix + Ky * Hiy + Kz * Hiz) / K^2 * ...
    % (Fx11 * Kx * Hix + Fx12 * (Ky*Hix + Kx*Hiy) + Fx13 * (Kz*Hix + kx*Hiz) + Fx22 * Ky*Hiy + Fx23 * (Kz*Hiy + Ky*Hiz) + Fx33 * Kz*Hiz)

    % F*chi vector : [Fx11, Fx12, Fx13, Fx22, Fx23, Fx33]
    
    Fx = reshape(in, [params.SS,6]);
    Res = zeros([params.SS, params.N_direction]);
    
    for n = 1:params.N_direction
        H_Vec = params.H_Matrix(n,:);
        kH_over_k2 = (H_Vec(1) * params.kx + H_Vec(2) * params.ky + H_Vec(3) * params.kz) ./ (eps + params.k2);
        
        Res(:,:,:,n) = ((H_Vec(1)^2)/3 - H_Vec(1)*params.kx .* kH_over_k2) .* Fx(:,:,:,1) + ...                         %   Fx11 
            (2*(H_Vec(1)*H_Vec(2))/3 - (H_Vec(1)*params.ky + H_Vec(2)*params.kx) .* kH_over_k2) .* Fx(:,:,:,2) + ...    %   Fx12
            (2*(H_Vec(1)*H_Vec(3))/3 - (H_Vec(1)*params.kz + H_Vec(3)*params.kx) .* kH_over_k2) .* Fx(:,:,:,3) + ...    %   Fx13
            ((H_Vec(2)^2)/3 - H_Vec(2)*params.ky .* kH_over_k2) .* Fx(:,:,:,4) + ...                                    %   Fx22
            (2*(H_Vec(2)*H_Vec(3))/3 - (H_Vec(2)*params.kz + H_Vec(3)*params.ky) .* kH_over_k2) .* Fx(:,:,:,5) + ...    %   Fx23
            ((H_Vec(3)^2)/3 - H_Vec(3)*params.kz .* kH_over_k2) .* Fx(:,:,:,6);                                         %   Fx33
    end
    
    res = Res(:);
    
    fprintf('+')
end

end

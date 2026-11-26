% MATLAB references for GIAnT-Python tests/raster/test_raster_trial_data.py
% (R2024b; nansvd.m is GIAnT-MATLAB main 5a0673b's dependencies/nansvd.m)
rng(7);
out = struct();

% smoothdata's moving mean: even, odd and fractional windows
x = [1 nan 3 4 nan nan 7 8 9 10 nan 12 2 5 nan 1];
w = [4 5 4.6 5.4 3.2];
out.mm_x = x; out.mm_w = w; out.mm_y = nan(numel(w), numel(x));
for k = 1:numel(w)
    out.mm_y(k, :) = smoothdata(x, 2, 'movmean', w(k), 'omitmissing');
end

% medfilt2 with NaN, and the labeled pixels of loadTrial
A = 10 * rand(9, 11); A(rand(size(A)) < 0.2) = nan;
out.med_in = A; out.med_out = medfilt2(A, [3 3]);
lab = medfilt2(A, [3 3]);
out.labeled = double(~isnan(A) & lab > 3 * prctile(lab(~isnan(lab)), 25));

% motion upsampling: interp1 pchip with extrapolation
ds = ceil(3:133.58:3000); mot = cumsum(randn(size(ds))) * 0.3;
lines = ceil(1:53.4333:3050);
out.pchip_x = ds; out.pchip_y = mot; out.pchip_q = lines;
out.pchip_v = interp1(ds, mot, lines, 'pchip', 'extrap');

% discarded frames at the extraction's frames
d = rand(1, 37) < 0.2; n = 91;
out.disc_in = double(d);
out.disc_out = double(interp1(1:numel(d), double(d(:)), linspace(1, numel(d), n)) > 0);
out.blocks = round(linspace(1, 5633 + 1, ceil(5633 / 600) + 1));

% nansvd and the denoised ROI trace of loadTrial
t = 1:300;
D = (rand(12, 1) + 0.5) * (5 + sin(t / 9)) + 0.4 * randn(12, 300) + 3 * rand(12, 1) * cos(t / 23);
D(rand(size(D)) < 0.05) = nan; D(3, :) = nan; D(:, 40:44) = nan; D(7, 1:150) = nan;
[UU, SS, VV, bg] = nansvd(D, 3, 10, 0.33);
roiLikeness = (abs(mean(UU, 1, 'omitnan')) ./ sqrt(mean(UU.^2, 1, 'omitnan'))) .* diag(SS)';
[~, selPC] = max(roiLikeness);
out.svd_d = D; out.svd_s = diag(SS)'; out.svd_bg = bg; out.svd_recon = UU * SS * VV';
out.svd_trace = mean(bg + (UU(:, selPC) * SS(selPC, selPC) * VV(:, selPC)'), 1, 'omitnan');
D1 = D(5, :); % a one-pixel ROI: fewer components than requested
[UU, SS, VV, bg] = nansvd(D1, 3, 10, 0.33);
roiLikeness = (abs(mean(UU, 1, 'omitnan')) ./ sqrt(mean(UU.^2, 1, 'omitnan'))) .* diag(SS)';
[~, selPC] = max(roiLikeness);
out.svd1_trace = mean(bg + (UU(:, selPC) * SS(selPC, selPC) * VV(:, selPC)'), 1, 'omitnan');

% the end of loadTrial (processAllTrials_Async.m, after the block loop):
% discards, aberrantly negative samples, gap filling, variance multiplier
IMsel = 5 + randn(6, 2, 120); Finvsel = 1 + rand(6, 120);
IMsel(2, :, :) = nan;               % a pixel without any sample
IMsel(3, 1, 20:75) = nan;           % a gap longer than the window
IMsel(4, 2, [1:5 100:120]) = nan;   % gaps at both ends
IMsel(5, 1, 60) = -40;              % an aberrantly negative sample
IMsel(6, :, 30:33) = nan;
Finvsel(squeeze(isnan(IMsel(:, 1, :)))) = nan;
discardFrames = false(1, 48); discardFrames([10 11 30]) = true;
params.baselineWindow_samps = 16;
out.fin_y_in = IMsel; out.fin_finv_in = Finvsel; out.fin_discard_in = double(discardFrames);
out.fin_window = params.baselineWindow_samps;
if numel(discardFrames) > 1
    discard = interp1(1:numel(discardFrames), double(discardFrames(:)), linspace(1, numel(discardFrames), size(IMsel,3)))>0;
else
    discard = repmat(any(discardFrames(:)), 1, size(IMsel,3));
end
IMsel(:,:,discard) = nan;
IMsel(IMsel<-2*std(IMsel,0,3, 'omitmissing')) = nan;
nans = isnan(IMsel);
nanFill = IMsel;
nanFill(all(nans,3)) = 0;
passes = 0;
while any(isnan(nanFill), 'all')
    nanFill = smoothdata(nanFill,3,"movmean",params.baselineWindow_samps, 'omitmissing');
    passes = passes + 1;
end
IMsel(nans) = nanFill(nans);
Finvsel(squeeze(nans(:,1,:))) = 1000*mean(Finvsel,'all', 'omitmissing');
out.fin_y = IMsel; out.fin_finv = Finvsel; out.fin_discard = double(discard); out.fin_passes = passes;
fprintf('passes %d\n', passes);

save('matlab_trial_data.mat', '-struct', 'out', '-v7');
disp(out.mm_y);

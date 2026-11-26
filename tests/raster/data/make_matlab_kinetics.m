function make_kinetics_fixture(out)
% Reference values of islocalmax and sourceKinetics/rejectFastSources for
% GIAnT-Python's tests (GIAnT-MATLAB main 5a0673b, read-only).
% Writes a v7.3 .mat; make_matlab_kinetics_h5.py turns it into the test's
% matlab_kinetics.h5:
%   make_kinetics_fixture('kinetics_fixture.mat')   (dependencies on the path)
%   python make_matlab_kinetics_h5.py kinetics_fixture.mat matlab_kinetics.h5
rng(7);
% islocalmax with plateaus and MinSeparation
Y = randi([0 5], 300, 40);
Y(1:100, :) = Y(1:100, :) + 0.001 * randn(100, 40);
seps = [0 1 2 3 5 9.6];
M = false(numel(seps), size(Y,1), size(Y,2));
for i = 1:numel(seps)
    M(i, :, :) = islocalmax(Y, 2, 'MinSeparation', seps(i));
end
% kinetics: trials with sensor-like and fast transients, discards
params = struct('tau_s', 0.024, 'analyzeHz', 200, 'minKineticScore', 0.2);
nS = 10; nT = 600; nTr = 3;
tauF = params.tau_s * params.analyzeHz;
E = cell(nTr, 1);
LS = nan(nTr, nS, nT);
for t = 1:nTr
    x = 0.3 * randn(nS, nT) + 0.05 * (1:nT) / nT;
    for s = 1:nS
        on = find(rand(1, nT) < 0.02);
        for p = on
            a = 3 + 3 * rand;
            if s <= 4
                k = exp(-(0:min(40, nT - p)) / tauF);   % sensor kinetics
            elseif s <= 7
                k = [1 0.3];                            % fast transients
                k = k(1:min(2, nT - p + 1));
            else
                k = exp(-(0:min(40, nT - p)) / (tauF / 3)); % between
            end
            x(s, p:p + numel(k) - 1) = x(s, p:p + numel(k) - 1) + a * k;
        end
    end
    discard = rand(1, nT) < 0.05;
    discard(100:120) = true;
    x(:, discard) = nan;
    if t == 2
        x(9, :) = nan;      % a source without data in one trial
    end
    E{t}.dF.ls = cat(3, x, randn(nS, nT));
    LS(t, :, :) = x;
end
E{nTr + 1} = [];            % a trial without traces
[score, nEvents] = sourceKinetics(E, params);
sources = struct('R', (1:nS)', 'C', (2:nS + 1)', 'V', ones(nS, 1));
[~, kept, kin] = rejectFastSources(E, sources, params);
keep = ismember((1:nS)', kept.R);
save(out, 'Y', 'seps', 'M', 'LS', 'score', 'nEvents', 'keep', 'kin', '-v7.3');
end

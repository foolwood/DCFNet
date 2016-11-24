function kf = linear_correlation(xf, yf)
kf = sum(xf .* conj(yf), 3) / numel(xf);
end
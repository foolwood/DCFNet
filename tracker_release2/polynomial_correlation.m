function kf = polynomial_correlation(xf, yf, a, b)
%POLYNOMIAL_CORRELATION Polynomial Kernel at all shifts, i.e. kernel correlation.
%   Evaluates a polynomial kernel with constant A and exponent B, for all
%   relative shifts between input images XF and YF, which must both be MxN.
%   They must also be periodic (ie., pre-processed with a cosine window).
%   The result is an MxN map of responses.
%
%   Inputs and output are all in the Fourier domain.
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/
	
	%cross-correlation term in Fourier domain
	xyf = xf .* conj(yf);
	xy = sum(real(ifft2(xyf)), 3);  %to spatial domain
	
	%calculate polynomial response for all positions, then go back to the
	%Fourier domain
	kf = fft2((xy / numel(xf) + a) .^ b);

end


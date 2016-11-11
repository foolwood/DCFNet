function test_ResponseLossSmoothL1()

window_sz = [125,125];
target_sz = [50,50];
sigma = sqrt(prod(target_sz))/10;

net = dagnn.DagNN() ;

ResponseLossSmoothL1 = dagnn.ResponseLossSmoothL1('win_size', window_sz,'sigma',sigma) ;
net.addLayer('ResponseLossSmoothL1', ResponseLossSmoothL1, {'r','delta_yx'}, {'loss'}) ;

r(:,:,1,1) = gaussian_shaped_labels(sigma, window_sz);
r(:,:,1,2) = circshift(r(:,:,1,1),[10,10]);
r(:,:,1,3) = circshift(r(:,:,1,1),[20,20]);
r(:,:,1,4) = circshift(r(:,:,1,1),[30,30]);

net.eval({'r',r,'delta_xy',[10,10;10,10;10,10;10,10]});
response = net.vars(net.getVarIndex('loss')).value ;
subplot(2,2,1);imagesc(squeeze(response(:,:,1,1)));title('( 0, 0)ResponseLossSmoothL1');colorbar();
subplot(2,2,2);imagesc(squeeze(response(:,:,1,2)));title('(10,10)ResponseLossSmoothL1');colorbar();
subplot(2,2,3);imagesc(squeeze(response(:,:,1,3)));title('(20,20)ResponseLossSmoothL1');colorbar();
subplot(2,2,4);imagesc(squeeze(response(:,:,1,4)));title('(30,30)ResponseLossSmoothL1');colorbar();

end

function labels = gaussian_shaped_labels(sigma, sz)%kcf
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));
labels = circshift(labels, -floor(sz(1:2) / 2) + 1);
assert(labels(1,1) == 1)
end

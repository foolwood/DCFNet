function test_delta2response()

window_sz = [125,125];
target_sz = [50,50];
sigma = sqrt(prod(target_sz))/10;

net = dagnn.DagNN() ;

delta2response = dagnn.delta2response('win_size', window_sz,'sigma',sigma) ;
net.addLayer('delta2response', delta2response, {'delta_xy'}, {'idea_response'}) ;

net.eval({'delta_xy',[10,10;10,-10;-10,10;-10,-10]});
response = net.vars(net.getVarIndex('idea_response')).value ;
subplot(2,2,1);imagesc(squeeze(response(:,:,:,1)));title('(10,10)idea response');
subplot(2,2,2);imagesc(squeeze(response(:,:,:,2)));title('(10,-10)idea response');
subplot(2,2,3);imagesc(squeeze(response(:,:,:,3)));title('(-10,10)idea response');
subplot(2,2,4);imagesc(squeeze(response(:,:,:,4)));title('(-10,-10)idea response');

end

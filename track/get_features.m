function x = get_features(im, features, cell_size, cos_window, net)
%GET_FEATURES
%   Extracts dense features from image.
%
%   X = GET_FEATURES(IM, FEATURES, CELL_SIZE)
%   Extracts features specified in struct FEATURES, from image IM. The
%   features should be densely sampled, in cells or intervals of CELL_SIZE.
%   The output has size [height in cells, width in cells, features].
%
%   To specify HOG features, set field 'hog' to true, and
%   'hog_orientations' to the number of bins.
%
%   To experiment with other features simply add them to this function
%   and include any needed parameters in the FEATURES struct. To allow
%   combinations of features, stack them with x = cat(3, x, new_feat).
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/


if features.hog,
    %HOG features, from Piotr's Toolbox
    x = double(fhog(single(im) / 255, cell_size, features.hog_orientations));
    x(:,:,end) = [];  %remove all-zeros channel ("truncation feature")
end

if features.gray,
    %gray-level (scalar feature)
    x = double(im) / 255;
    
    x = x - mean(x(:));
end

if features.alex,
    im_ = single(im) ;
    im_ = bsxfun(@minus, im_, single(reshape([123,117,104],[1,1,3]))) ;
    res = vl_simplenn(net, im_) ;
    x = squeeze(gather(res(end).x)) ;
    sz = size(x);
    x_col = reshape(x,[],sz(3));
    mn = mean(x_col);
    sd = std(x_col);
    sd(sd==0) = 1;
    mn = reshaple(mn,[1,1,sz(3)]);
    sd = reshaple(sd,[1,1,sz(3)]);
    x = bsxfun(@minus,x,mn);
    x = bsxfun(@rdivide,x,sd);
    
    % for i = 1:size(x,3)
    %     x1 = x(:,:,i);
    %     mn = mean(x1(:));
    %     sd = std(x1(:));
    %     sd(sd==0) = 1;
    %     xn = bsxfun(@minus,x1,mn);
    %     xn = bsxfun(@rdivide,xn,sd);
    %     x(:,:,i) = xn;
    % end
end

if features.vgg,
    im_ = single(im) ;
    im_ = bsxfun(@minus, im_, single(reshape([123,117,104],[1,1,3]))) ;
    res = vl_simplenn(net, im_) ;
    x = squeeze(gather(res(end).x)) ;
    
    sz = size(x);
    x_col = reshape(x,[],sz(3));
    mn = mean(x_col);
    sd = std(x_col);
    sd(sd==0) = 1;
    mn = reshape(mn,[1,1,sz(3)]);
    sd = reshape(sd,[1,1,sz(3)]);
    x = bsxfun(@minus,x,mn);
    x = bsxfun(@rdivide,x,sd);
    
% for i = 1:size(x,3)
%     x1 = x(:,:,i);
%     mn = mean(x1(:));
%     sd = std(x1(:));
%     sd(sd==0) = 1;
%     xn = bsxfun(@minus,x1,mn);
%     xn = bsxfun(@rdivide,xn,sd);
%     x(:,:,i) = xn;
% end

end

%process with cosine window if needed
if ~isempty(cos_window),
    x = bsxfun(@times, x, cos_window);
end

end

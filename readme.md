# DeepKCF

首先，KCF+HOG特征取得了非常好的速度与精度的平衡，在此基础上发展了CF2,DeepSRDCF等CNN特征加上KCF的方法，但是他们都是直接采用vgg在imagenet库上训练好的模型，这些模型并不一定适应于跟踪问题。

那么我们仔细分析，如果只是考虑一种编码问题，或者特征空间的投影，hog实际上是对每个cell进行了一个编码。对于KCF来说是一个4*4的cell，产生了31维的的编码，在这个编码的基础上进行了DCF运算。（不考虑kernel trick）。

那么CF2使用了cf2的多层特征进行融合。

```
indLayers = [37, 28, 19];   % The CNN layers Conv5-4, Conv4-4, and Conv3-4 in VGG Net

nweights  = [1, 0.5, 0.02]; % Weights for combining correlation filter responses
```
对应的感受野分别是[];
其最终的结果为0.891 0.605
相对于KCF的提升还是很显著的。但是问题是，这样的特征是实验出来的，跑了很多参数自然不必说，那么最好的方法当时是对应KCF的end-to-end的训练，同时，注意到，Siamese-fc其实就是一个end-to-end的很好的事例，但是缺陷是最好一级的ncc。本身cnn就是位置无关的，ncc对每个位置也是均衡加权，所以不是很好，所以作者用一个高斯窗口进行加权，但还是没有kcf这样的优化解（回归问题）好。所以我认为cnn+kcf的end-to-end训练是最好的一种解。
另外，KCF框架也可以作为网络的一个重要部件，因为他不仅在最后一段产生了在线学习（注意是超快速的在线学习，学习效果并不比svm弱），另外一个重要的原因是，KCF能带来时序信息，并且不增加额外的计算，想象一下，LSTM可以做最后一集合（ROLO），但是随着step的增加，速度必然下降。虽然效果好，那你还是一个短时的滑窗。并且ROLO的很要命的弱点是，他是针对人，或者训练好的物体，这不符合常规的单目标的假定。原因就是，要想搞定通用物体的LSTM，你需要无比巨大的数据，我在训练GOTURN的时候，就直接对这种回归的方法产生了一定的抵触。


下面进入我的实验思考环节。第一步，如何在KCF的基础上改进呢。Martin Danelljan已经把理论部分做的没太多可做了，大家也灌水不少，那还是从特征出发吧，毕竟我这个工作想的是end-to-end kcf。那么一上来要直接重写个kcf层么，先别这么宏远，我们先来**学习hog**。

这非常的重要，我想明白直接cnn的特征不work，或者还有最优，到底什么样的才是最优。既然开山鼻祖给我们留下了hog，只有[4*4]的感受野，31个feature map。可以产生这么好的性能，我觉得首先肯定值得学习hog特征。鉴于在归一化的时候还是用了临近的一个cell，所以，最多的感受野也就是8，那么我么就直接上[3*3]的小卷积核，[3-3-31],这样最简单粗暴，但是显然是不对的，非线性跑哪里去了。那就按照vgg的框架来吧。下面是实验具体细节。

## Train

### Experiment1 （hog）
实验做的有一定的问题，简单是vgg前三层学习起来总会产生0输出，非常奇怪。



今天搜索到了这个文章，里面非常厉害的直接用CNN生成HOG。帮我省去了大烦恼，其实自己也应该学着写的。这样。网络模型的初始模型就有了。

网络是个浅层网络，这也验证了，我们可以使用一个浅层的网络来进行跟踪。没必要上十几层的VGG之类的。并且，假设CNN-HOG是一个局部最优解，那么我们可以使用数据灌溉到更优解。希望顺着这个思路能有收货。

Understanding Deep Image Representations by Inverting Them



### Experiment2 （dcf layer）
继续把MOSSE的文章的推倒，加上KCF的框架仔细看了看。现在的初步计划是使用DCF。
理由1，DCF使用的是线性核，速度会比高斯核快很多。
理由2，性能上并不比高斯核的弱很多(差一个百分比的样子，这样一来，我们有理由相信，Kernel trick只是一个trick，真正好使的还是特征，也对，你只有一个初始目标，这样的内积空间能映射的有多好)
先从dcf layer开始写起。

前项测试了应该没有大问题。直观上vgg对于局部的编码还是要远好与gray的直接输入的。
#### gray+DCF的结果
明显可以看出，在很多地方产生了噪声
[gray_dcf](/train2/gray_dcf.pdf)
#### vgg16+DCF的结果
[vgg_dcf](/train2/vgg_dcf.pdf)
预测的响应更接近目标状态，当然，完全重现理想状态一定会产生过拟合，因为有各种变化，如果要是简单的平移，理论上是可以得到非常近似的结果。
那么我们要设计一个怎样的损失呢。
最大值很多时候都是正确的，问题是有时候会产生以下干扰的尖刺，在MOSSE中认为是遮挡，剧烈变化等问题造成的，扰动会产生PSR下降是很容易理解的，那么是否可以找到有效的编码，内在的去除抖动，我觉得这个可以作为网络学习的一部分。此外，如何去除远处的无干扰，是kcf模型需要急切解决的，我觉得应该惩罚次峰值。
类似于SRDCF，如果响应在距离中心位置较远处得到了较高的响应应该被惩罚。但是，旁边的小平坦没有比较被强行降低到0.所以打算如下的损失：

```
sum((idea_respone-respone).*(respone>(0.1*max(respone(:) ) ) ).*...
(1-gaussian2d(should located) ))

```

一步步分析，

```
respone>(0.1*max(respone(:) ) ) 
```
如果比最大响应的0.1还小的区域不需要被关注，这些解无论是什么都不会影响最终的结果。需要被修正的是潜在干扰项，或者错误项。

```
(1-gaussian2d(should located) )
```
这就是SRDCF的正则系数，但是我们用这个来修正约束损失，那么如果在远离真正的目标中心位置处产生了损失会受到更为严重的惩罚。
只有采取l1还是别的差值，这个只能做实验看了。


Spectral Representations for Convolutional Neural Networks
这篇文章有讲一下如何对频域的求导，这样的解答也让我对运算产生了信心。

下面进一步推导整个过程
##### Forward

1.x -> F(x)		R[h,w,c,n]->C[h,w,c,n]

```
xf = fft2(x);
```

2.F(x)-> F(kxx)		C[h,w,c,n]->C[h,w,1,n] 我现在还是不知道为什么要除以xf

```
kxxf = sum(xf.*conj(xf),3)/ numel(xf);
```

3.F(kxx)+F(y)-> F(alpha)		C[h,w,1,n]->C[h,w,1,n]

```
alphaf = yf./(kxxf+lambda);
```

4.F(z)+F(x)-> F(kzx)		C[h,w,c,n]->C[h,w,1,n]

```
kzxf = sum(zf.*conj(xf),3)/ numel(xf);
```

5.F(alpha)+F(kzx)-> r(response)			C[h,w,1,n]->R[h,w,1,n]

```
r = real(ifft2(alphaf.*kzxf));
```

##### Backward

1.dl/dr ->dl/drf		R[h,w,1,n]->C[h,w,1,n]

```
dldrf = fft2(dldr);
```

2.dl/drf ->	dl/dkzxf		C[h,w,1,n]->C[h,w,1,n]

```
dldkzxf = dldrf.*alphaf;
```

3.dl/dkzxf ->dl/dzf		C[h,w,1,n]->C[h,w,c,n]

```
dldzf = bsxfun(times,dldkzxf./ numel(xf),conj(xf));
```

4.dl/dzf ->	dl/dz		C[h,w,c,n]->R[h,w,c,n]

```
dldzf = ifft2(dldzf);
```

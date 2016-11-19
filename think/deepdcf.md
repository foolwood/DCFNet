#DeepDCF

首先还是先回顾DCF，对后面所需要用到的公式先进行理论上的推倒。并且同时梳理一下相关的论文。
MOSSE,CSK,KCF,CN(color names),DSST,SRDCF,DeepSRDCF,CF2.

##Discriminative Correlation Filters

###DSST
**(PCA-HOG+Gray)+DCF+Scale Esimation**

key word: **circular correlation**,**Parserval's identity**, **dense feature** 

patch: 
$$
f_ {1},...,f_ {t}
$$ 
label: 
$$
g_ {1},...,g_ {t}
$$
filter: 
$$
h_ {t}
$$
test patch: 
$$
z
$$

$$
\epsilon =  \sum_{j=1}^{t}\left \| h_{t}\star f_{j}-g_{j} \right \|^{2} = \sum_{j=1}^{t}\left \| \overline{H}_{t} F_{j}-G_{j} \right \|^{2}
$$


$$
H_{t}=\frac{\sum_{j=1}^{t}\overline{G}_{j}F_{j}}{\sum_{j=1}^{t}\overline{F_{j}}F_{j}} 
$$

$$
y=\mathfrak{F}^{-1}{\overline{H}_{t}Z}
$$

那么这是最简单的单特征的情况。
对于Multidimensional Features情况稍微复杂一点。尤其是文中指出要是像上面那样优化一系列时间1,...,t的话，需要dxd个线性方程组，复杂度过高，所以只能对每一个时刻单独求。然后做线性加权的近似。

patch: 
$$ 
f^{l} ,l\in \{ 1,...,d \} 
$$
label: 
$$
g
$$
filter: 
$$
h^{l}
$$
test patch: 
$$
z^{l}
$$

$$
\epsilon =  \sum_{l=1}^{d}\left \| h^{l}\star f^{l}-g \right \|^{2} +\lambda \sum_{l=1}^{d}\left\|h^{l} \right\|^{2}
$$

$$
H^{l}=\frac{\overline{G}F^{l}}{\sum_{k=1}^{d}\overline{F^{k}}F^{k}+\lambda}
$$


T obtain a robust approximation, here we update the numerator $$ A_ {t}^{l} $$ and denominator $$ B_{t} $$

$$
A_{t}^{l} = (1-\eta )A_{t-1}^{l}+\eta \overline{G}_{t}F_{t}^{l}
$$

$$
B_{t}^{l} = (1-\eta )B_{t-1}^{l}+\eta \sum_{k=1}^{d}\overline{F}_{t}F_{t}^{l}
$$

响应计算的结果
$$
y=\mathfrak{F}^{-1}\left\{ \frac{\sum_{l=1}^{d}\overline{A^{l}}Z^{l}}{B+\lambda}\right\}
$$


###CN


###SRDCF







##DeepDCF
将**CNN**网络看成是一种非线性映射。将原始rgb（或灰度扩展）图像进行映射。
$$
\Phi(x)=f^{l}
$$
$$
\Phi(z)=z^{l}
$$

patch: 
$$
x^{l} ,l\in \{ 1,...,d \} 
$$
label: 
$$
g
$$
filter: 
$$
h^{l}
$$
test patch: 
$$
z^{l}
$$

$$
\epsilon =  \sum_{l=1}^{d}\left \| h^{l}\star x^{l}-g \right \|^{2} +\lambda \sum_{l=1}^{d}\left\|h^{l} \right\|^{2}
$$

$$
H^{l}=\frac{\overline{G}X^{l}}{\sum_{k=1}^{d}\overline{X^{k}}X^{k}+\lambda}
$$

响应计算的结果
$$
y=\mathfrak{F}^{-1}\left\{ \frac{\sum_{l=1}^{d}\overline{A^{l}}Z^{l}}{B+\lambda}\right\}
$$

其实到这了这，和上面的DSST实际上是一样的。因为我本身就是利用的DCF，公式没有任何变化，只是利用闭式解反传误差。下面为了更方便写代码(forward和backward)：
进行整合的推倒：

$$
y=\mathfrak{F}^{-1}\left\{ \frac{\sum_{l=1}^{d}\overline{A^{l}}Z^{l}}{B+\lambda}\right\}
=\mathfrak{F}^{-1}\left\{\frac{ \sum_{k=1}^{d}\overline{\overline{G}X^{l}}Z^{l}}{\sum_{k=1}^{d}\overline{X^{k}}X^{k}+\lambda}\right\}
=\mathfrak{F}^{-1}\left\{\frac{ \sum_{k=1}^{d}(Z^{l}\overline{X^{l}})G}{\sum_{k=1}^{d}\overline{X^{k}}X^{k}+\lambda}\right\}
$$

```
y=ifft2((sum3(fft2(z).*conj(fft2(X))).*fft2(g))./(sum3(fft2(X).*conj(fft2(X)))+lambda)));
```

前项过程相对来说较为容易，反向就有点恶心了。
已知
$$
\frac{\partial l}{\partial y}
$$
求
$$
\frac{\partial l}{\partial z} , \frac{\partial l}{\partial x}
$$

那么还是先用公式证明：
公理1，
$$
Y=\mathfrak{F}\left\{y\right\},\frac{\partial l}{\partial Y} = \mathfrak{F}\left\{\frac{\partial l}{\partial y}\right\}
$$
公理2，
$$
\frac{\partial f(x,x^{*})}{\partial x}=\overline{\frac{\partial f(x,x^{*})}{\partial x^{*}}}
$$
且易知
$$
Y_{uv}=\frac{\sum_{k=1}^{d}(Z_{uv}^{l}\overline{X_{uv}^{l}})G_{uv}}{\sum_{k=1}^{d}X_{uv}^{k}\overline{X_{uv}^{k}}+\lambda}
$$

那么
$$
\frac{\partial l}{\partial Z_{uv}^{l}}=\frac{\partial l}{\partial Y_{u,v}}\frac{\partial Y_{u,v}}{\partial Z_{uv}}=\mathfrak{F}\left\{\frac{\partial l}{\partial y}\right\}_{uv}  \frac{\overline{X_{uv}^{l}}G_{uv}}{\sum_{k=1}^{d}X_{uv}^{k}\overline{X_{uv}^{k}}+\lambda}
$$

$$
\frac{\partial l}{\partial X_{uv}^{l}}
=\frac{\partial l}{\partial Y_{u,v}}\frac{\partial Y_{u,v}}{\partial X_{uv}}
=\mathfrak{F}\left\{\frac{\partial l}{\partial y}\right\}_{uv}
\frac{
\overline{Z_{uv}^{l}G_{uv}}(\sum_{k=1}^{d}\overline{X_{uv}^{k}}X_{uv}^{k}+\lambda)-\overline{X_{uv}^{k}}(\sum_{l=1}^{d}(Z_{uv}^{l}\overline{X_{uv}^{l}})G_{uv})}
{(\sum_{k=1}^{d}\overline{X_{uv}^{k}}X_{uv}^{k}+\lambda)^{2}}
$$



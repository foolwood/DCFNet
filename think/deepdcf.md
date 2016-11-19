#DeepDCF

首先还是先回顾DCF，对后面所需要用到的公式先进行理论上的推倒。并且同时梳理一下相关的论文。
MOSSE,CSK,KCF,CN,DSST,SRDCF,DeepSRDCF,CF2.

##Discriminative Correlation Filters

###DSST
**(PCA-HOG+Gray)+DCF+Scale Esimation**

key word: **circular correlation**,**Parserval's identity**, **dense feature** 

patch: 
$$ f_ {1},...,f_ {t} $$ 
label: 
$$ g_ {1},...,g_ {t} $$
filter: 
$$ h_ {t} $$
test patch: 
$$ z $$

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
$$ f^{l} ,l\in \left \{ 1,...,d \right \} $$
label: 
$$ g $$
filter: 
$$ h^{l} $$
test patch: 
$$ z^{l} $$

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







# Expectation-Maximization-for-Gaussian-Mixture-Model
Gaussian Mixture Model (GMM)을 활용한 Probabilistic Density Function(PDF) estimation 및 Expectation Maximization (EM) 알고리즘


## 1. GMM Algorithm
Gaussian Mixture Model (GMM)은 이름 그대로 Gaussian 분포가 여러 개 혼합된 clustering 알고리즘이다. 현실에 존재하는 복잡한 형태의 확률 분포를 [그림 1]과 같이 KK개의 Gaussian distribution을 혼합하여 표현하자는 것이 GMM의 기본 아이디어이다. 이때 KK는 데이터를 분석하고자 하는 사람이 직접 설정해야 한다.

![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/%EA%B7%B8%EB%A6%BC1.png)

[그림 1] 여러 Gaussian distribution의 혼합 분포

주어진 데이터 xx에 대해 GMM은 xx가 발생할 확률을 [식 1]과 같이 여러 Gaussian probability density function의 합으로 표현한다.

![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/%EA%B7%B8%EB%A6%BC2.png)

[식 1]에서 mixing coefficient라고 하는 πk는 k번째 Gaussian distribution이 선택될 확률을 나타낸다. 따라서, πk는 아래의 두 조건을 만족해야 한다.

![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/%EA%B7%B8%EB%A6%BC3.png)

![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/%EA%B7%B8%EB%A6%BC4.png)

GMM을 학습시킨다는 것은 주어진 데이터 X={x1,x2,...,xN}X={x1,x2,...,xN}에 대하여 적절한 πk, μk, ,Σk를 추정하는 것과 같다.

## 2. EM Algorithm

기댓값 최대화 알고리즘(expectation-maximization algorithm, 약자 EM 알고리즘)은 관측되지 않는 잠재변수에 의존하는 확률 모델에서 최대가능도 (maximum likelihood)나 최대사후확률 (maximum a posteriori, 약자 MAP)을 갖는 모수의 추정값을 찾는 반복적인 알고리즘이다. EM 알고리즘은 모수에 관한 추정값으로 로그 가능도 (log likelihood)의 기댓값을 계산하는 기댓값 (E) 단계와 이 기댓값을 최대화하는 모수 추정값들을 구하는 최대화 (M) 단계를 번갈아가면서 적용한다. 최대화 단계에서 계산한 변수값은 다음 기댓값 단계의 추정값으로 쓰인다.

## 3. Experiment Way

본 실험의 개발 환경은 다음과 같다.

Python 3.7.6

본 실험에서 주어진 좌표(Points.csv)를 통해 먼저 Means(평균)과 Covariance(공분산, Sigma라고도 한다.)을 구하고자 했다.

![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/Figure_1.png)

그 결과 위와 같은 그래프를 얻으며, 다음과 같은 값을 얻었다.

Priors (사전 확률)
[0.27232708 0.27546473 0.45220819]

means (평균)
[[69.20722901 19.70652909]
 [29.84835715 80.14776865]
 [79.80717341 69.65192526]]
 
covars (공분산)
[[85.2867814   3.3551158 ]
 [ 3.3551158  84.26381303]]
[[115.31568863   0.72129237]
 [  0.72129237  85.71009274]]
[[216.90027901 129.13394185]
 [129.13394185 211.39333946]]

3가지의 값을 얻었는데, Priors(잠재변수 z 에 대한 사전지식이 사전확률), means(평균), 그리고 covariance(공분산)이다.

이제 이를 통해, EM 알고리즘을 이용해 GMM 모델을 학습하여 PDF를 생성하기로 한다.
### 실험 1
![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/gaussian%202d.png)

![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/Complete.png)

EM 알고리즘에서 매개 변수 집합을 초기화하여, 확률 추정과 변수 집합 추정을 특정 조건까지 반복합니다. 이를 통해 초기화 과정에서 실험을 진행해보기로 했다. 먼저 실험 1 위의 그래프는 초기화 변수 집합을 목표 평균과 공분산을 그대로 넣어 훈련을 진행했다. Iteration이 4도 안된 채 최적해에 도달한 모습이다. 이를 앞으로는 원본이라고 부르겠다.

### 실험 2

pis = [1 / 4, 1 / 4, 1 / 2]

mus = np.array([[23, 11], [23, 18], [11, 21]])

sigmas = [[85.2867814, 3.3551158], [3.3551158, 84.26381303]], [[115.31568863, 0.72129237],
 [0.72129237 , 85.71009274]], [[216.90027901, 129.13394185], [129.13394185 ,211.39333946]]

위와 같은 변수 집합이 있다고 가정하자. 이는 공분산은 똑같이 하고, 평균과 사전확률을 바꾼 값이다. 이를 알고리즘 모델에 넣어서 계속 돌리면 과연 우리가 원하고자 하는 모양에 가까울 수 있을까?

![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/gmm_test1.png)
![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/gmm_test1%202.png)
![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/gmm_test1%203.png)
![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/gmm_test1%203.png)

첫 시작 이후, iteration 2에서 급격한 변화를 얻었다. 이후 iteration 199가 될 때까지 쭉 같은 모양의 그래프가 지속됐다. 초기 해에서 가장 많은 변화를 줬던 세번째 평균 값이 올바르게 진행을 하다가 두 개의 군집에서 넘어가지 못하는 것 같았다. 이는 최적해로 수렴하지만, Greedy Algorithm인 EM이 초기 해에 따라 최종해가 달라지며, 전역 최적해가 아닌 지역 최적해로 수렴할 수 있다는 점을 보여준다.

![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/gmm_test1%202d.png)

신기한 점은 군집 역시도 2개로 생성된다는 점이다. 이에 대해서 연구가 많이 필요하겠지만 군집을 이루지 못한 위쪽의 좌표들이 한 군집으로 훈련된 것으로 보인다.

### 실험 3 

pis = [1 / 4, 1 / 4, 1 / 2]

mus = np.array([[69.20722901, 19.70652909],
[29.84835715, 80.14776865],
[79.80717341, 69.65192526]])

sigmas = [[88, 23], [33, 84]], [[11, 23.72],
 [15.7212 , 82.71009274]], [[22, 111.13394185], [16.133 ,200.39333946]]
 
 다음은 위와 같이 평균을 원본과 유지하고, 공분산과 사전확률을 바꿔 훈련시켜 보았다.
 
 ![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/gmm_test3%20%20start.png)
 ![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/gmm_test3%20%201.png)
 ![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/gmm_test3%20%202.png)
 
 첫 시작지점에서 iteration 0이 되자, 원본과 같은 모양을 가면서, iteration 2부터 iteration 25까지는 원본과 거의 유사한 모습을 보이고 있다. 
 
 ![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/gmm_test3%202d.png)
 
 2D로 평면화한 데이터에서도 원본과 흡사한 데이터를 볼 수 있었다. 이를 통해 평균의 변화가 공분산의 변화보다 큰 영향을 끼친다는 것을 알 수 있었다. 물론 평균점의 위치가 변화한다면 당연히 첫 시작 지점에 따라 훈련은 달라질 수 있다.
 
 ### 실험 4
 
pis = [1 / 4, 1 / 4, 1 / 2]

mus = np.array([[123, 121], [23, 63], [21, 31]])

sigmas = [[88, 63], [33, 34]], [[131, 23.72],
 [0.7212 , 52.71009274]], [[202, 111.13394185], [126.133 ,200.39333946]]
 
다음은 모든 데이터를 바꾸어 훈련을 해보았다. 너무 크게 변화를 주면 훈련이 진행되지 않을 수도 있어서, 적당히 바꿔보도록 했다.

![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/gmm_test4%20start.png)
![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/gmm_test4%201.png)
![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/gmm_test4%202.png)
![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/gmm_test4%203.png)
![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/gmm_test4%204.png)
![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/gmm_test4%205.png)

![image](https://github.com/Junst/Expectation-Maximization-for-Gaussian-Mixture-Model/blob/main/images/gmm_test4%202d.png)

첫 시작 지점이 서로 멀리 떨어져 있었지만, iteration 2에서 급격히 왼쪽으로 치우친 그래프가 나타났다. 이후 점점 오른쪽으로 가면서, 큰 군집 하나를 만들어내고, 시간이 감에 따라 두 군집으로 분화하여 원본과 매우 닮은 그래프가 나타났다. 이를 통해 알 수 있었던 점은, 첫 시작 위치가 다른 군집 좌표로부터 서로 멀어지는 것은 최적해에 도달할 가능성이 크다는 점이다. 실험 2와 다른 점은 한 군집을 원본의 두 군집 사이에 둬서 지역적 최적해에 도달하여 군집을 못이룬 실험 2와 다르게 실험 4는 가려는 방향에 다른 군집이 없어서 흡수되거나 지역적 최적해에 도달하지 않고 전역 최적해에 근접하여 나왔다는 점이다. 또한 이를 통해 실험 2에서 세웠던 가설인 평균의 변화가 공분산의 변화보다 더 큰 영향을 준다는 가설은 그 근거가 약해진 느낌이 있다. 즉, 시작 위치가 중요하다는 점이다. 이러한 가설들의 신뢰성을 높이기 위해선 다양한 근거와 실험 연구가 필요할 것이다.

## 4. Conclusion & Discussion
본 연구의 결과는 다음과 같다.

훈련하려는 데이터의 첫 위치(초기화)가 훈련과 군집화에 영향을 크게 준다. 또한 그 과정에서 평균의 변화가 공분산의 변화보다 더 큰 변화를 줄 수도 있다는 결과를 얻기도 하였다.

그러나, 본 실험은 다음과 같은 한계점이 있다.

첫 째로, 실험에서 세웠던 가설을 채택할 만큼 충분한 근거 및 표본이 부족하다는 점이 있다.

둘 째로, 해당 실험에서 쓰였던 코드에서 평균과 공분산을 구하기 위한 코드와 그 이후에 실험을 진행한 코드가 다르다는 점이다.

더 나아가 이러한 코드가 분리되어 진행된 점을 통해 연구자의 깊이 있는 연구와 해당 분야에 대한 지식이 더 필요한 것으로 보인다는 점이다. 본 연구는 해당 연구에서 바꿀 수 있는 변수들을 통해 GMM 모델을 최대한 이해하고자 하였다.






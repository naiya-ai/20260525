학습된 모델에 대해 다음과 같은 평가를 한다.

분석 space는 명시적으로 지정한다. 현재 beta=0.1 best model 분석에서는
`gaussian_quantile` transformed space를 사용한다.

주의: decoded conditional prior mean은 별도 지표/그림으로 만들지 않는다.
아래 분석은 conditional prior에서 latent code를 sampling한 뒤 decode한 generated value만 사용한다.

1. test sample들의 각 numerical disease indicator에 대해 x축에 true value, y축에 prior sampled latent code를 decoding한 generated value를 scatter로 그린다.

2. test sample들을 각 indicator순으로 정렬한 후, top 10%, top 10~20%, ..., 90~100% 들을 분리하여 conditional prior sampling으로 indicator를 generate한다. hist로 distribution을 그린다. 비교를 위해 최상단에 data distribution을 함께 그린다.

3. 각 indicator top 10% sample 중 4개를 선택하여 각각 N=1000개의 prior sampling을 하여 distribution을 그린다. true value를 표시한다. 비교를 위해 최상단에 data distribution을 함께 그린다.

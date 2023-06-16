# [DACON] 제주도 도로 교통량 예측 AI 경진대회 [(Link)](https://dacon.io/competitions/official/235985/leaderboard)

## 🏆 Result
- **Public score 2nd** 3.07714 | **Private score 4th** 3.0852 | **최종 3등(🏆)**




주최 : 제주 테크노파크, 제주특별자치도

주관 : DACON

규모 : 총 1400여명 참가

===========================================================================




<div align=center>
  
  ![단락 텍스트](./img/img.png)
</div>


✏️
**대회 느낀점**
 - 실제 도로에서 나타나는 현상(ex, 러시아워)을 적용하는 과정에서 외부 데이터를 많이 사용하며 성능을 끌어올려 제공된 데이터 뿐만 아닌 외부의 많은 데이터를 융합하는 시야를 넓혔고, 실제 도로에서 과속이나 차량 사고 시 도로 막힘 현상에 대해
   고려를 해야 하는것이 맞나, 맞지 않나를 생각하며 다양한 이상치 처리 방법(3sigma, IQR)들을 적용시켜 보았고 이런 경험을 통해 데이터에 나타나는 여러가지 이상 현상에 대해 생각하는 자세를 가지게 되었습니다.
   
   또한 대회 중 수 많은 피처 중에 "Lane count" 즉 차선 수 피처가 존재했었는데, 차선 별로 나누어서 모델링을 했던 적이 있었고 예측 모델의 성능이 잘 나왔습니다.
   하지만 이전 대회에서 Data-Leakage 이슈가 있었기 때문에 이러한 모델링 방법이 맞는지에 대해 생각을 하게 되었고,
   train 데이터셋에는 차선이 1,2,3 차선 종류만 존재했었습니다. 하지만 train 데이터셋에 존재하지않는 차선 수, 예를 들어 5차선, 10차선이 test 데이터셋에 존재하면 어떻게 되는거지? 라는 생각이 들어.
   이 방법은 Data-Leakage 라 판단을 하고 이러한 방법론을 과감하게 버린 후 새로운 방법으로 모델링을 진행하였습니다.
   대회가 종료된 후 저희가 버렸던 방법론인 차선 별로 모델링을 진행한 팀이 존재하였고, 결국 그 팀은 2차 평가에서 Data-Leakage로 실격처리 되었습니다.

   또한 외부데이터를 사용하며 카카오맵 API를 연동하여 방문자수, 리뷰수가 많은 구간들은 사람들이 많이 몰리는 **Hot Place** 구간으로 그 주변 도로들의 평균 속도가 전체적으로 낮아질 것이라는 가설을 세우고 모델에 적용을 시켰습니다.
   이 방법을 사용하였을 때 예측 성능이 가장 많이 향상 되었지만 카카오맵API는 실시간으로 업데이트가 되기 때문에 test셋인 2022년 8월 데이터를 포함하게 됩니다.
   그래서 최종적으로 카카오맵 API를 사용하는 방법도 버리게 되었지만, 실제 end-to-end 모델을 개발하여 실시간으로 예측할 때는 많은 도움이 될 것이라 생각하여 주최사측에 의견 전달을 하였습니다.
   
   이러한 경험을 통해 현재 제공된 데이터셋(train)에서 어떻게 test 데이터를 보지 않고 일반화 성능을 올리는지에 대해 많은 생각을 할 수 있었고 앞으로도 이러한 이슈에는 유연하게 잘 대응을 할 수 있을 것입니다.

===========================================================================

스태킹 앙상블 등 다양한 기법들을 시도하여 성능을 최대한으로 끌어올려보려 했지만
Data-Leakage에 위반이 되지 않는 것이 더 중요하다 생각이 들어 마지막 시간은 코드 점검에만 시간을 쏟았습니다.

다양한 모델링 기법을 제대로 해보지 않았기 때문에 성능 향상 가능성이 높은 코드라 생각합니다.

Data-Leakage에 위험하다 생각되는 코드들은 마지막에 모두 삭제하였으며 최종 제출 점수도 가장 높은 PB Score가 아닌 위험하다 생각한 피처들을 모두 삭제하고 학습, 예측한 안전한 PB Score로 제출 하여 Public Score 2등에서 Private Score 4등으로 마무리하였습니다.

===========================================================================


## Data Info

* 각각의 칼럼에 대한 설명
* Features 
  * 아이디 / 날짜 / 요일 / 시간대 / 차로 수 / 도로 등급 / 중용구간 여부 / 연결로 코드 / 최고 속도 제한 / 통과 제한 하중 / 통과 제한 높이 / 도로 유형 / 시작 지점 위도 / 시작 지점 경도 / 도착 지점 위도 / 도착 지점 경도 / 시작 지점 회전 제한 여부 / 도착 지점 회전 제힌 여부 / 도로 명 / 시작 지점 명 / 도작 지점 명 / 통과 제한 차량
* Target
  * 평균 속도 (KM)
  
### Train Set

* Rows : 4,701,217 개
* Columns : 24 개
  * float64 : 9 개
  * int64 : 8 개
  * object : 7개

### Test Set

* Rows : 291,241 개
* Columns : 23 개
  * float64 : 8 개
  * int64 : 8 개
  * object : 7개
  
---

## Process

**1. EDA** 
  * 평균 도로 속도가 40인 도로의 평균 속도가 40을 초과하는 제일 높은 속도임을 확인, 데이터 불균형 확인
  * 2022년 7월 이전과 이후로 평균 target값의 차이가 심함을 파악
  * 각 도로의 특징 별로 target값의 차이가 심함을 파악
  * 각 도로마다 target의 이상값이 존재 -> 사고 or 폭주로 생각
  * 요일, 시간대별로 target 값 차이 존재
  
**2. Feature Engineering** 
  * 시간 feature들은 inherently cyclical 하다는 특징을 활용하기 위해 sin/cos 변환
  * 특정 피처들의 target 값의 평균으로 파생 변수 생성
  * 위,경도 값을 기준으로 clustering 하여 파생 변수 생성 -> 특정 도로는 인접한 도로읙 교통량에 영향을 미칠것이란 가설
  * 제주도 관광지 외부데이터를 활용해 위,경도 값을 기준으로 특정 거리 내의 관광지들을 count하는 파생 변수 생성 -> 관광지가 많은 곳은 교통량이 혼잡할 것이다 라는 가설
  * 공휴일 외부데이터를 활용해 공휴일 당일, 전,후날 파생변수 생성 -> 공휴일에는 교통량이 많을 것이라는 가설
  * 이외의 feature engineering을 통해 총 48개의 피처로 학습 진행

**3. Modeling**

  * XGBoostRegressor 모델 사용
  * Optuna를 활용하여 하이퍼파라미터 튜닝
  * 교차 검증 활용(stratified k-fold cross validation)
  * target 값은 정수형으로 이루어져있다는 사실에 기반하여 예측값을 round 

**추가 시도 사항** (최종 결과 도출에는 비사용)
  * IQR * 1.5 2.0 등등 값을 변화해가며 폭주 or 사고라고 생각하는 이상치들을 drop -> 정확도가 오히려 낮아짐
  * 3 sigma 법칙을 이용해 폭주 or 사고라고 생각하는 이상치들을 drop -> 정확도가 오히려 낮아짐
  * Stacking 기법 -> 마지막에 시도하였지만 시간이 부족해 base model의 충분한 하이퍼파라미터 튜닝을 하지 못해 성능이 약간 하락
    -> 대회 종료 후 다시 확인하였더니 Stacking 기법이 더 점수가 좋았음
  * 관광지 관련 파생변수 생성할 때 카카오맵 API를 사용하였는데 DataLeakage로 최종 제출에는 사용하지 않음 -> 제일 성능이 좋았음
  * 다양한 KFold K 변경


  

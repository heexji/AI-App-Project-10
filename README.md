@@ -1,28 +1,48 @@
# 인공지능응용 10팀 - 프로젝트 repository
광운대학교 인공지능응용 수업의 팀 프로젝트 자료를 정리한 공간입니다.
프로젝트는 아래 논문을 기반으로 구현 및 실험을 진행합니다.

## 📄 참고 논문
**LDAM: Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss**
https://arxiv.org/abs/1906.07413

## 👥 팀원
- 김선규
- 김희지
- 배상윤
- 정석영
- 조희문

## 📌 프로젝트 개요
본 프로젝트의 목표는 논문 기반 모델 재현 및 비교실험입니다.

- 데이터 전처리 및 실험 파이프라인 구축

- 모델 학습 및 성능 비교

- 결과 해석 및 시각화

## 실행방법
pip install -r requirements.txt(추후 추가 예정)

cifar100 데이터 분석법

아래의 분석코드 실행시 자동으로 불균형 cifar 데이터셋 생성
python cifar_train.py `
>>   --gpu 0 `
>>   --dataset cifar100 ` ##미입력시 기본값 cifar10
>>   --imb_type exp `  ##불균형 데이터 타입 exp밖에 구현 안되어있음
>>   --imb_factor 0.01 ` ##불균형 팩터 디폴트 0.01
>>   --loss_type CE ` ##손실함수 타입, 기본값 CE(cross entropy), Focal, LDAM 구현되어있음
>>   --train_rule None ` ##불균형 학습 방법. Resample, Reweight, DRW 구현되어있음



bdd weather 데이터 분석법

download_bdd.py 실행 후 설치 경로 확인

이후에 마찬가지로 아래 실행
python cifar_train.py `
>>   --gpu 0 `
>>   --dataset bdd_weather `
>>   --imb_type exp `
>>   --imb_factor 0.01 `
>>   --loss_type CE `
>>   --train_rule None `
>>   --bdd_root "C:\Users\(username)\.cache\kagglehub\datasets\solesensei\solesensei_bdd100k\versions\2" ##설치된 실제 경로 확인


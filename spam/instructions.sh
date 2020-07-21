# 로그인

nsml login

#기본 돌리는 키

nsml run -e nsml_train.py -d spam-1 -m "A good message" -g 1 -a "--experiment_name v1"

# 이 부분에서의 arguement에 대한 설명을 조금 하면
# -e는 entry를 의미하고, -d는 데이터셋, -g는 할당할 gpu 개수, -a 가 python file의 arguments를 의미한다.

# 로그 출력

nsml logs nill1024/spam-1/ #이후 번호

# 모델 조회 

nsml model ls nill1024/spam-1/~

# 모델 삭제

nsml model rm # checkpoint를 기준으로 없앤다. 이건 documentation 한번 더 보는걸 추천함

# 세션 모두 보기

nsml ps -a

# submit (--test로 세션 테스트)

nsml submit nill1024/spam-1/~ 

# nsml pip install command

pip install git+https://github.com/n-CLAIR/nsml-local


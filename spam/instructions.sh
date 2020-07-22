# 로그인

nsml login

#기본 돌리는 키

nsml run -e nsml_train.py -d spam-1 -g 1 -a "--experiment_name v1"

# 이 부분에서의 arguement에 대한 설명을 조금 하면
# -e는 entry를 의미하고, -d는 데이터셋, -g는 할당할 gpu 개수, -a 가 python file의 arguments를 의미한다.
# -i, --interactive 세션의 훈련이 다 끝난 뒤에 세션은 자동으로 삭제됨
# -c, --cpus string_or_integer 사용할 cpu 갯수입니다. ex) nsml run -c 5


# 로그 출력

nsml logs nill1024/spam-1/ #이후 번호

# 모델 조회 

nsml model ls nill1024/spam-1/~

# 모델 삭제

nsml model rm # checkpoint를 기준으로 없앤다. 이건 documentation 한번 더 보는걸 추천함

# 세션 삭제 -f 같은 경우에 submit test가 좀비 상태가 되었을때 없애는 용도로 씀

nsml rm -f [sessionID]

# 세션 모두 보기

nsml ps -a

# submit (--test로 세션 테스트)

nsml submit nill1024/spam-1/~ 
#--test로 print 로그를 확인할 수 있다고 함.


# nsml pip install command

pip install git+https://github.com/n-CLAIR/nsml-local

# nsml run 하였을 때 upload 된 session 파일들을 local machine 으로 가져온다고 하는데 정확히 뭔지 모르겠음

nsml pull -v nsmlteam/mnist/4 ./


# nsml fork 새로운 세션으로 재생성 이라고 되어 있음
# 기본적으로 run과 같은 인자를 사용하는데, --checkpoint 있음 (checkpoint가 -c 인자를 가져감)

nsml fork nill1024/spam-2/2 -g 5 
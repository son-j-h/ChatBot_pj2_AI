### Toy project[LangChain/SpringBoot] 
2025.07 -
# 챗봇 웹페이지 제작
## 5인 팀프로젝트 역할 분담
● 나현지| agent구현/훈련장려금 지급 핸들러/파이프라인 메시지 출력/답변 피드백 기능<br/>
● 남현수| 출결 신청·조회·취소 자동 처리 핸들러/도커 환경 구축<br/>
● 박서원| 증명서 발급 핸들러/아키텍처 구성/DB서버 구축 및 관리/웹 구현<br/>
● 손장호| 공가·휴가 사용 핸들러/핸들러 퓨샷 기능 추가<br/>
● 주영경| 출결 정정 핸들러/비속어 필터링<br/>

## 프로젝트 소개
패스트캠퍼스 부트캠프 수강생들의 행정문의 및 관리를 위한 LLM 챗봇<br/>
<img width="696" height="470" alt="Image" src="https://github.com/user-attachments/assets/49fbb915-de8e-4428-bcdf-913ab0b6720d" /><br/>
<img width="1157" height="586" alt="Image" src="https://github.com/user-attachments/assets/a4f5d1d8-80da-4bba-8070-fe17ae652852" /><br/>
<img width="1153" height="448" alt="Image" src="https://github.com/user-attachments/assets/25b6a40e-403a-4857-b284-acc0a1a57050" /><br/>

## 기술스택
● 백엔드 프레임워크: SpringBoot<br/>
● AI 서버 연결: Flask<br/>
● AI 모델 및 프레임워크 : OpenAI + LangChain (LLM 응답 흐름 제어 및 프롬프트 체인 구성) => GoogleGenerativeAI로 변경 *토큰 비용 문제 <br/>
● 시스템 운영 : Docker<br/>

### 아키텍처 구성<br/>
<img width="1146" height="573" alt="Image" src="https://github.com/user-attachments/assets/9db91e4a-0266-44b2-a92b-841bb9be30a3" /><br/>

### 데이터베이스 설계<br/>
<img width="1185" height="532" alt="Image" src="https://github.com/user-attachments/assets/57d686d7-43da-4919-8e4d-ef918e0e4022" /><br/>

### 웹 구현 부분
웹(스프링부트): https://github.com/Seowon-Park/ChatBot_pj2<br/>

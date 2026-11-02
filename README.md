# codemason 3.0

**codemason 은 같은 구현을 두 언어로 유지한다.** 이 저장소는 **Rust 판**이고,
Python 판은 [AmesianX/codemason-py](https://github.com/AmesianX/codemason-py) 에 있다.
한쪽을 고치면 다른 쪽에도 반영한다 — 기능과 동작은 서로 맞춰 간다.

> ## ⚠️ 고지 · Disclaimer
>
> 이 저장소는 **있는 그대로(as-is)** 공개됩니다. 어떤 보증도 하지 않습니다.
>
> - 이 코드를 사용해 발생하는 **모든 결과와 책임은 사용자 본인**에게 있습니다.
> - 이 도구는 파일을 읽고 **고치고 지우며, 셸 명령을 실행합니다.** 무엇을 실행할지는
>   모델이 정합니다. 중요한 작업 트리에서는 쓰지 마시고, 버전 관리 아래에서만 쓰세요.
> - 관할 **법령과 제3자 약관(모델·API·서비스 라이선스 포함)을 지킬 책임은 사용자**에게 있습니다.
>
> *Provided **as-is**, without warranty of any kind. This tool edits files and executes
> shell commands under model control. See [LICENSE](LICENSE).*

터미널에서 도는 코딩 에이전트 CLI. **로컬 LLM**(vLLM / llama.cpp)을 상대로 돌며
클라우드에 의존하지 않는다.

```
153개 .rs 파일 / 73,871줄 · 테스트 609개 · Rust 2021 edition
```

## 설치

```bash
./setup.sh                 # 리눅스·macOS — 빌드 후 실행 파일과 vendor/ 를 함께 배치
setup.bat                  # 윈도우 — 설치 경로를 유저 PATH 에 등록 (새 터미널부터)
cargo build --release      # 직접 빌드할 때
```

실행 파일만 다른 곳으로 복사하면 동봉된 ripgrep 을 잃고 느린 네이티브 워크로
떨어진다. `setup.sh` / `setup.bat` 이 둘을 같이 옮기는 이유다.

## 사용

```bash
codemason                  # REPL — 타이핑은 큐잉되고 작업 중에도 입력을 받는다
codemason -p "..."         # 프린트 모드. 한 번 묻고 결과만 출력
codemason --help
```

모델 백엔드는 설정으로 고른다 (`vllm` 권장 / `llamacpp`). OpenAI 호환 엔드포인트를 쓴다.

## 실행 엔진 권장 사항

> **llama.cpp 기반 실행은 구현되어 있으나, 성능과 구현 완성도를 고려하여 사용하지
> 않을 것을 강력히 권고합니다. 본 프로젝트에서는 vLLM 사용을 권장합니다.**

현재 본 프로젝트의 b12x 커널은 vLLM에만 구현되어 있으며, 프로젝트 실행 환경에서
prefill 1,300+ 토큰/초, decoding 35~40+ 토큰/초의 성능으로 쾌적한 사용이 가능합니다.
반면 llama.cpp는 최대한 최적화한 후에도 prefill 450+ 토큰/초, decoding 25~30 토큰/초
수준에 머물러, 체감 성능에서 상당한 차이를 보였습니다.

또한 사용한 llama.cpp 구성에서는 2노드 텐서 병렬이 기본적으로 지원되지 않아 이를 직접
구현하고 추가 최적화까지 진행했으나, 여전히 만족할 만한 체감 성능을 확보하지 못했습니다.

이러한 이유로 현재 개발과 최적화는 vLLM에 집중하고 있습니다. llama.cpp는 기본적인 실행
기능만 구현된 상태로, 추가 기능 구현과 충분한 테스트가 이루어지지 않았으므로 사용을
권장하지 않습니다.

## 구성

```
src/
├── main.rs            진입점
├── cli.rs             REPL UI — 입력 엔진과 출력 라우팅
├── cli_repl.rs        REPL 루프
├── query.rs           에이전트 루프 — 모델 호출, 도구 실행, 자동 압축,
│                      최대 토큰 복구, 반복 호출 제동
├── print_runner.rs    -p 모드 실행기
├── task.rs / tasks/   백그라운드 작업
├── tools/             Read · Write · Edit · Bash · Glob · Grep · Agent ·
│                      WebFetch · MCP · TodoWrite · AskUserQuestion …
├── services/          백엔드 어댑터 (vllm, llamacpp), 압축, 비전, A/B
├── remote/            원격 봇 풀
├── state/             세션 · 재개 · 트랜스크립트
├── constants/         프롬프트와 스키마
├── utils/             설정, 파일, 포맷, 권한 상승, 완성, 암호화 …
└── locales/           다국어 (영어 원문이 키)
```

### 동작 개요

1. **입력** — REPL 이 입력을 큐에 넣는다. 모델이 도는 중에도 타이핑이 막히지 않고,
   다음 턴에 steering 으로 들어간다.
2. **에이전트 루프** (`query.rs`) — 모델을 부르고, 돌려받은 도구 호출을 실행하고,
   결과를 다시 넣는다. 대화가 길어지면 자동으로 압축하고, 같은 호출이 반복되면 제동을 건다.
3. **도구** (`tools/`) — 각 도구는 스키마와 실행부를 함께 가진다. 파일 수정은
   읽기를 선행 조건으로 두고, 셸은 권한 확인을 거친다.
4. **상태** (`state/`) — 세션을 트랜스크립트로 남겨 재개할 수 있게 한다.

## 라이선스

MIT. [LICENSE](LICENSE) 참조.

Claude Code 의 동작을 참고해 독자 구현했으며, 원 제품과 제휴 관계가 없다.

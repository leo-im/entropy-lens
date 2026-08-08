# entropy-lens

**한국어** | [English](README.en.md)

**LLM logprobs로부터 토큰 단위 엔트로피 궤적을 — logprobs in, entropy trajectory out.**

`entropy-lens`는 LLM API가 이미 반환하는 `logprobs`만으로 토큰별 불확실성
궤적을 계산·시각화합니다. 다중 샘플링도, 보조 모델도, 추가 forward pass도
필요 없기 때문에 서빙 파이프라인에서 모든 요청에 대해 돌려도 될 만큼
가볍습니다.

![CoT 엔트로피 궤적](docs/assets/demo_trajectory.png)

*데모 플롯은 구조화된 합성 데이터로 생성한 것입니다(`python scripts/generate_demo.py`);
`scripts/verify_trajectory.py`를 사용하면 실제 모델로부터 같은 플롯을 얻을 수 있습니다.*

## 설치

```bash
pip install entropy-lens            # core (numpy만 의존)
pip install "entropy-lens[viz]"     # + matplotlib 시각화
```

## Quickstart

OpenAI 호환 클라이언트(vLLM, SGLang, OpenAI)로 logprobs를 요청하고, 응답을
그대로 `entropy-lens`에 넘기면 됩니다:

```python
from openai import OpenAI
from entropy_lens.adapters import from_openai_response
from entropy_lens.viz import plot_trajectory

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "What is 12 * 3 - 7? Think step by step."}],
    logprobs=True,
    top_logprobs=20,
)
traj = from_openai_response(response, split="sentence")
print(traj.summary())  # 평균/최대/최소 H, 스텝별 감소량, 단조성
plot_trajectory(traj)  # 토큰 단위 + 스텝 단위 엔트로피 플롯
```

`traj`는 `EntropyTrajectory`입니다: 토큰별 엔트로피(`traj.entropies`, 기본
단위는 bits라서 perplexity가 `2**H`로 해석됩니다), 토큰 문자열, 그리고 스텝
경계와 스텝 단위 통계(`step_means()`, `delta_h()`, `summary()`)를 담습니다.

## 왜 엔트로피 궤적인가?

스칼라 하나짜리 confidence는 생성 과정의 *어디가* 불확실했는지 숨깁니다.
궤적 관점 — 토큰별 엔트로피를 추론 스텝 단위로 집계 — 은 디코딩 중
불확실성의 구조를 드러냅니다:

- **분기점**: 엔트로피 스파이크는 여러 continuation이 실제로 그럴듯했던
  위치를 표시합니다. CoT 응답에서는 추론 경로가 갈라지는 지점입니다.
- **수렴 패턴**: 전형적인 다단계 추론에서는 제약이 쌓이면서 스텝별 평균
  엔트로피가 감소하는 경향이 있고, 이 패턴에서 벗어나는 지점이 유의미한
  신호가 됩니다.
- **서빙 친화적 측정**: 모든 것이 단일 pass의 logprobs에서 나오므로, 디코딩
  비용을 바꾸지 않고도 요청별 텔레메트리, retrieval-stopping 정책(예: 기대
  정보 이득(EIG) 기준), 추론 품질 진단의 기반으로 쓸 수 있습니다.

`entropy-lens`는 의도적으로 *측정* 계층에 머뭅니다: 그 위에 올라갈
정보이론 기반 reliability 도구들의 공통 기반으로 설계되었습니다.

**파인튜닝 진단에도 쓸 수 있습니다.** 학습(LoRA/SFT) 중 에폭마다 고정
프로브 세트에 `generate()`를 돌려 궤적을 추적하면 entropy collapse나
calibration 변화 같은 현상을 관찰할 수 있습니다 — "학습 중 측정"도 결국
추론 시점 측정이므로 기존 어댑터(`from_hf_generate`)로 그대로 됩니다.
학습 루프·실험 코드는 이 라이브러리의 범위 밖이며, 별도 실험 레포
에서
이 라이브러리를 의존성으로 사용합니다.

## 연구 철학

이 프로젝트가 속한 연구는 다음 질문에서 출발합니다: *지능적인 reasoning이란
현재의 불확실성을 파악하고, 가장 informative한 evidence를 획득하며,
hypothesis space를 올바른 방향으로 점진적으로 제약하여 reliable decision에
도달하는 과정으로 볼 수 있는가?*

이 프로젝트의 답: **그렇다 — 단, "올바른 방향"인지는 시스템 스스로 완전히
알 수 없다.** 불확실성의 크기·고유성·수렴은 시스템이 스스로 잴 수 있지만,
방향(truth alignment)의 최종 보증은 시스템 바깥 — gold reference 또는 인간
판단 — 에서 온다. 따라서 신뢰할 수 있는 지능 시스템의 설계 문제는 "완벽한
자기 평가"가 아니라 **"유한한 외부 검증 자원을 자기 신호로 최적 배분하는
것"**이며, 이것이 정보이론(불확실성 정량화)과 인식론(진실의 근거)과 시스템
공학(human-in-the-loop)이 만나는 지점이다.

`entropy-lens`는 이 그림에서 "시스템이 스스로 잴 수 있는 것" — 단일 pass
logprobs로부터의 불확실성 궤적 — 을 담당하는 측정 계층입니다.

## API 한눈에 보기

| 함수 | 역할 |
| --- | --- |
| `token_entropy(logprobs, base="bits", tail="ignore")` | 한 위치의 top-k logprobs로부터 Shannon entropy |
| `sequence_entropies(logprobs_per_token, ...)` | 시퀀스 전체의 토큰별 엔트로피 배열 |
| `perplexity_from_entropy(H)` | `2**H` — 그럴듯한 다음 토큰의 유효 개수 |
| `split_steps(tokens, "sentence" \| "paragraph" \| "step_marker", pattern=...)` | 토큰을 추론 스텝으로 분할 |
| `EntropyTrajectory` | `step_means()`, `delta_h()`, `summary()`, 플로팅 입력 |
| `adapters.from_openai_response(resp)` | chat/legacy completions logprobs 파싱 (OpenAI/vLLM/SGLang) |
| `adapters.hf_transformers.from_hf_generate(out, tok)` | HF `generate()` scores로부터 전체 vocab 정확 엔트로피 |
| `viz.plot_trajectory(traj)` | 토큰 단위 + 스텝 단위 엔트로피 figure |

## 한계

**top-k logprobs로 계산한 엔트로피는 *추정치*이지 참값이 아닙니다.** API는
위치당 top-k(보통 ≤20)개의 logprobs만 반환합니다. `tail="ignore"`(기본값)는
top-k 질량을 재정규화하며 실제로는 참 엔트로피를 과소평가하는 경향이
있고(하한 성격의 추정), `tail="uniform"`은 잔여 질량을 나머지 vocabulary에
균등 분배해(`vocab_size` 필요) 상한 성격의 추정을 제공합니다. 참값은 두
추정치 사이에 있으며, 분포가 집중되거나 k가 커질수록 둘은 수렴합니다.
HuggingFace 어댑터는 전체 vocabulary를 보기 때문에 이 한계가 없습니다.

**낮은 엔트로피 ≠ 정답.** 엔트로피는 모델의 *확신*을 측정할 뿐이고, 모델은
확신에 차서 틀릴 수 있습니다(특히 RLHF 계열 튜닝 이후에는 calibration이
깨져 있는 경우가 많습니다). 궤적이 하강한다는 것은 불확실성이 붕괴했다는
뜻이지 답이 맞다는 뜻이 아닙니다. `entropy-lens`는 측정 도구이지 정확도
판단 도구가 아닙니다.

**토크나이저 의존성.** 엔트로피는 서빙 모델의 토큰에 붙는 값입니다.
토크나이저가 다른 모델 간에 절대값을 비교하는 것은 주의 없이는 의미가
없습니다.

## 검증 스크립트

각 마일스톤마다 사람이 직접 확인할 수 있는 검증 스크립트를 제공합니다
(`scripts/` 참고):

```bash
python scripts/verify_math.py                              # 수학적 sanity check
python scripts/verify_adapter.py tests/fixtures/vllm_response.json   # token | H | ppl 테이블
# 라이브 서버 필요 (vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-logprobs 20):
python scripts/verify_live.py --base-url http://localhost:8000/v1        # 저 vs 고 엔트로피 대조 (스모크)
python scripts/verify_battery.py --base-url http://localhost:8000/v1     # 대조 12쌍 통계 배터리 (승률 >= 80%)
python scripts/verify_trajectory.py --base-url http://localhost:8000/v1  # CoT 플롯 -> verify_output/
```

GPU가 없다면 `verify_live.py`는 OpenAI API로도 동작합니다(`--base-url
https://api.openai.com/v1 --api-key ... --model gpt-4o-mini`).

## 예제

- [`examples/01_basic_trajectory.ipynb`](examples/01_basic_trajectory.ipynb) — fixture 응답 → 토큰별 엔트로피 테이블과 플롯
- [`examples/02_cot_step_entropy.ipynb`](examples/02_cot_step_entropy.ipynb) — CoT 스텝 분할, ΔH, 스텝 단위 궤적

## 인용

이 소프트웨어를 연구에 사용하셨다면 아래와 같이 인용해 주세요
(GitHub의 "Cite this repository" 버튼으로도 복사할 수 있습니다):

```bibtex
@software{lim2026entropylens,
  author  = {Lim, Seungmin},
  title   = {entropy-lens: Token-level entropy trajectories from LLM logprobs},
  year    = {2026},
  url     = {https://github.com/leo-im/entropy-lens},
  version = {0.1.0}
}
```

## 라이선스

MIT

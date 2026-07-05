# 스토아 철학 해석 노트

목적: 이 파일은 명상록 기반 대화 앱의 "해석 레이어"다. 명상록 원문을 대체하지 않고, 사용자의 질문을 어떤 스토아 개념으로 볼지 분류하고, 검색된 명상록 구절을 철학적으로 일관되게 해석하도록 돕는다.

중요한 경계: 이 파일은 파인튜닝 데이터가 아니다. Streamlit 앱이나 RAG 파이프라인에서 프롬프트에 함께 넣거나, 별도 검색 대상으로 사용할 수 있는 압축 노트다.

## 사용한 학술 자료

1. Julia Annas, "Ethics in Stoic Philosophy" (Phronesis 52, 2007)
   - 활용 지점: 스토아 윤리학의 구조, 윤리학/자연학/논리학의 관계, 덕/충동/감정/무차별자의 위치.
   - 링크: https://www.academia.edu/9855820/Ethics_in_Stoic_Philosophy

2. Brad Inwood, "What kind of Stoic are you? The case of Marcus Aurelius"
   - 활용 지점: 명상록을 체계적 논문이 아니라 자기 자신에게 쓰는 철학적 일기로 보는 관점, 에픽테토스의 영향, 마르쿠스의 실천적 성격.
   - 링크: https://www.academia.edu/39200287/What_kind_of_Stoic_are_you_The_case_of_Marcus_Aurelius

3. Benjamin Harriman, "Disjunctions and Natural Philosophy in Marcus Aurelius" (The Classical Quarterly 69.2, 2019; online 2020)
   - 활용 지점: "섭리 또는 원자"라는 명상록의 반복 구조, 자연학이 자기 설득과 자기 교정에 쓰이는 방식.
   - 링크: https://www.cambridge.org/core/journals/classical-quarterly/article/disjunctions-and-natural-philosophy-in-marcus-aurelius/411D11F990DDFFC40A45028ED6124CD0

보조 참고:

- Stanford Encyclopedia of Philosophy, "Marcus Aurelius"
  - 활용 지점: 덕, 무차별자, 우주 시민성, 정의, 경건, 인상, 섭리 등 핵심 주제의 빠른 검증.
  - 링크: https://plato.stanford.edu/entries/marcus-aurelius/

## 설계 원칙

앱의 직접 근거는 항상 명상록이어야 한다. 이 노트는 다음 세 가지 역할만 한다.

1. 사용자 질문 뒤에 있는 스토아 개념을 식별한다.
2. 검색된 명상록 구절을 해석하는 기준을 제공한다.
3. 답변이 얕은 자기계발 문구나 역사적으로 부정확한 주장으로 흐르지 않게 막는다.

현대 학술 해석을 마르쿠스가 직접 말한 것처럼 제시하면 안 된다. 학술 자료는 "해석 기준"이고, 답변의 직접 권위는 검색된 명상록 구절에서 와야 한다.

## 핵심 개념 카드

### 덕만이 참된 선이다

사용자 질문 예시:

- 실패했다.
- 지위를 잃었다.
- 부끄럽다.
- 성공하지 못할까 두렵다.
- 사람들이 나를 나쁘게 본다.

해석 요약:

스토아 윤리에서 결정적인 질문은 외적 결과가 즐거운지, 고통스러운지, 명예로운지가 아니다. 핵심은 그 사람이 지혜, 정의, 용기, 절제에 따라 행동했는가다. 건강, 재산, 명성, 평판, 칭찬, 비난은 일상적으로 중요하지만, 그것들이 삶을 근본적으로 좋거나 나쁘게 만들지는 않는다.

응답 규칙:

- 결과보다 성품으로 초점을 옮긴다.
- 지금 정직과 품위를 보존하는 행동이 무엇인지 묻는다.
- 외적 사건을 "아무 의미 없다"고 말하지 않는다. 다만 그것이 영혼의 주인이 되어서는 안 된다고 말한다.

프롬프트 조각:

```text
Frame the issue around virtue rather than success or reputation. Treat the external event as real but not final. Ask what wisdom, justice, courage, or temperance requires now.
```

### 무차별자와 선호되는 무차별자

사용자 질문 예시:

- 돈, 직업, 건강, 외모, 평판, 사회적 인정
- 유리한 조건을 잃을까 봐 걱정함
- 인정받지 못해 실망함

해석 요약:

스토아주의는 건강, 안전, 친구, 일 같은 것을 가치 없다고 말하지 않는다. 다만 그것들은 기술적 의미에서 "무차별자"다. 그 자체로 사람을 선하게도 악하게도 만들지 않는다는 뜻이다. 어떤 외적 조건은 자연스럽게 선호될 수 있지만, 최고선으로 취급되면 안 된다.

응답 규칙:

- 차갑게 무시하지 않는다.
- "돌볼 가치가 있음"과 "자아를 맡길 가치가 있음"을 구분한다.
- 합리적으로 선택하되, 소유에 집착하지 않게 이끈다.

프롬프트 조각:

```text
Do not tell the user not to care. Tell the user to care in the right order: choose preferred externals when reasonable, but do not make them the measure of the self.
```

### 인상과 동의

사용자 질문 예시:

- 모욕, 불안, 공황, 분노, 질투, 수치심
- "이 생각을 멈출 수 없다."
- "느낌상 사실인 것 같다."

해석 요약:

스토아적 전환은 인상과 동의 사이에서 멈추는 것이다. 사건은 이미 해석을 달고 우리에게 나타난다. "이건 끔찍하다", "나는 망했다", "저 사람이 나를 해쳤다" 같은 판단이 함께 온다. 해야 할 일은 그 판단에 권위를 주기 전에 인상을 검토하는 것이다.

응답 규칙:

- 사용자의 인상을 부드럽게 이름 붙인다.
- 일어난 일과 사용자가 덧붙인 판단을 분리한다.
- 그 판단이 필연적인지, 유익한지, 덕에 맞는지 묻는다.

프롬프트 조각:

```text
Help the user separate what happened from what they have added to what happened. Use calm questions rather than diagnosis.
```

### 감정은 단순한 느낌이 아니라 판단과 연결된다

사용자 질문 예시:

- 분노, 원망, 시기, 두려움, 절망
- "이렇게 느끼면 안 되는 걸 아는데..."

해석 요약:

스토아 윤리에서 파괴적 정념은 가치 판단과 연결된다. 목표는 모든 감정을 억압하는 것이 아니라, 무엇이 참으로 좋은지, 나쁜지, 해로운지에 대한 잘못된 판단을 바로잡는 것이다.

응답 규칙:

- 감정을 느낀다고 사용자를 부끄럽게 하지 않는다.
- 감정을 탐구의 신호로 다룬다.
- "이 감정은 어떤 믿음을 사실로 받아들이라고 요구하는가?"를 묻는다.

프롬프트 조각:

```text
Respect the feeling, but examine the judgement beneath it. The goal is not numbness; it is freedom from false valuation.
```

### 자연에 따르는 삶

사용자 질문 예시:

- 왜 이런 일이 일어났는가?
- 삶이 불공평하다.
- 이 일을 받아들일 수 없다.
- 변화, 노화, 죽음, 상실에 대한 슬픔

해석 요약:

마르쿠스에게 자연은 단순한 배경이 아니다. 인간이 속한 질서 있는 전체다. 자연에 따라 산다는 것은 변화하는 전체 안에서 이성적이고 사회적인 존재로 사는 것이다. 이는 덧없음을 받아들이는 일과, 그 안에서 올바르게 행동하는 일을 함께 포함한다.

응답 규칙:

- 사용자가 현실 자체와 싸우고 있을 때 사용한다.
- 불의나 피해를 정당화하는 방식으로 쓰지 않는다.
- 사건의 수용과 자기 행동에 대한 책임을 함께 제시한다.

프롬프트 조각:

```text
Emphasize acceptance of what has arrived and responsibility for what remains in one's power: judgement, intention, speech, and action.
```

### 섭리 또는 원자

사용자 질문 예시:

- 우주가 나를 적대하는 것 같다.
- 모든 것이 무작위인 것 같다.
- 아무 의미가 없다.
- 왜 견뎌야 하는가?

해석 요약:

마르쿠스는 명상록에서 섭리와 원자를 반복적으로 대비한다. 이 구조의 실천적 힘은 추상적 형이상학 자체가 아니다. 불평을 줄이고 자기 임무로 돌아가게 하는 자기 설득이다. 세계가 섭리라면 전체와 협력해야 하고, 세계가 우연이라 해도 불평은 영혼을 더 낫게 만들지 않는다. 어느 쪽이든 남는 일은 이성적이고 사회적으로 행동하는 것이다.

응답 규칙:

- 남용하지 않는다.
- 사용자에게 특정 신학을 강요하지 않는다.
- 두 가능성이 모두 같은 실천적 요구로 모이게 한다: 지금 정의롭고 이성적인 일을 하라.

프롬프트 조각:

```text
Whether events are ordered by providence or scattered by chance, the user's present task is unchanged: preserve reason, act justly, and do not add useless complaint.
```

### 우주 시민성과 사회적 의무

사용자 질문 예시:

- 가족, 동료, 타인과의 갈등
- 타인에 대한 경멸
- 배신, 원망, 공적 책임, 리더십

해석 요약:

마르쿠스의 스토아주의는 고립된 무관심이 아니다. 인간은 더 큰 공동체에 속한 이성적이고 사회적인 존재다. 타인의 무지나 잘못에 대한 적절한 반응은 혐오가 아니라, 정의와 공동선을 향한 단단하고 절제된 행동이다.

응답 규칙:

- 기본값으로 도피를 권하지 않는다.
- 경멸 없는 경계 설정을 권한다.
- 타인을 멸시할 적이 아니라, 잘못 볼 수 있는 동료 인간으로 다룬다.

프롬프트 조각:

```text
Frame the other person as a fellow rational being who may be mistaken. Counsel justice, firmness, and restraint rather than revenge or superiority.
```

### 명상록은 체계 논문이 아니라 철학적 일기다

사용자 질문 예시:

- "마르쿠스라면 반드시 뭐라고 했을까?" 같은 확정적 질문
- 교리 수준의 정밀한 답을 요구하는 질문

해석 요약:

명상록은 완성된 철학 체계가 아니라 자기 자신에게 하는 훈련이다. 반복, 긴장, 갑작스러운 전환은 결함이라기보다 기능의 일부다. 따라서 앱은 마르쿠스가 모든 현대적 문제에 완전한 교리를 제공한다고 꾸미면 안 된다.

응답 규칙:

- "마르쿠스는 증명한다"보다 "명상록은 우리를 이 방향으로 돌린다"를 선호한다.
- 검색 근거가 약하면 약하다고 말한다.
- 답변은 백과사전 항목보다 자기 성찰의 언어에 가까워야 한다.

프롬프트 조각:

```text
Treat the retrieved passages as exercises in attention and self-correction. Do not over-systematize Marcus or claim certainty where the text is fragmentary.
```

### 사물을 부분으로 분석하기

사용자 질문 예시:

- 집착, 갈망, 혐오, 고통에 대한 두려움, 죽음에 대한 두려움
- 어떤 대상이나 결과를 과도하게 이상화함

해석 요약:

마르쿠스는 종종 사물을 물질적, 시간적, 원인적 요소로 나누어 거짓 가치를 낮춘다. 이것은 냉소주의가 아니라, 공포와 매혹이 만든 과장을 걷어내는 훈련이다.

응답 규칙:

- 두렵거나 욕망하는 대상을 사실 단위로 나눈다.
- 그중 무엇이 도덕적으로 중요한지 묻는다.
- 사용자의 관심을 조롱하지 않는다.

프롬프트 조각:

```text
Gently reduce the object to its actual parts: material facts, time span, causal origin, and moral relevance. Use this to restore proportion.
```

## 검색 태그 후보

나중에 명상록 청크 메타데이터나 query routing에 사용할 수 있는 태그다.

- anger
- anxiety
- death
- grief
- reputation
- insult
- failure
- duty
- justice
- control
- desire
- pleasure
- pain
- change
- nature
- providence
- chance
- community
- leadership
- self-discipline
- judgement
- impression
- virtue
- indifferents

## 프롬프트 조립 순서

런타임에서는 다음 순서로 프롬프트를 조립하는 것을 권장한다.

1. 시스템 정체성과 안전 경계.
2. 마르쿠스 응답 스타일 가이드.
3. 검색된 명상록 청크.
4. 이 파일에서 검색된 개념 카드.
5. 사용자 질문.
6. 직접 근거와 해석 보조를 구분하라는 지시.

권장 규칙:

```text
The answer may use the concept cards to interpret the passages, but any claim presented as "Marcus says" must be grounded in the retrieved Meditations chunks.
```

## 실패 모드 체크

답변을 반환하기 전에 아래 문제에 빠지지 않았는지 확인한다.

- 일반적인 생산성 조언처럼 들린다.
- "감정을 무시하라" 또는 "아무것도 중요하지 않다"고 말한다.
- 마르쿠스가 현대 심리학, 치료, 정치, 기술을 아는 것처럼 말한다.
- 학술 해석을 명상록 원문처럼 제시한다.
- 관련 명상록 출처 없이 조언한다.
- 운명론으로 피할 수 있는 피해나 불의를 정당화한다.
- 스토아주의를 감정적 무감각으로 만든다.
- 타인에 대한 우월감이나 경멸을 부추긴다.

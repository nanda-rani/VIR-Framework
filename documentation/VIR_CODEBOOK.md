# Victim Interpretation and Response (VIR) Codebook

## Purpose and unit of analysis

VIR describes information explicitly expressed in one cybercrime complaint
narrative. It characterizes how a victim represents and responds to an
incident. It does not determine whether an allegation is true, assign legal
responsibility, diagnose a victim, or assess whether the victim behaved
correctly.

Annotators should use only evidence in the narrative. If evidence is missing or
ambiguous, choose the least specific supported label. Never infer a label from
demographics, crime category, presumed intent, or outside information.

## Interpretation dimensions

### 1. Perceived attack vector (`attack_vector`)

Single-label nominal field describing the pathway through which the narrator
believes the incident occurred.

| Code | Label | Use when |
|---:|---|---|
| 0 | Phishing email | The described initiating pathway is a deceptive email. |
| 1 | Phishing SMS/call | The pathway is a deceptive SMS, messaging interaction, or telephone call. |
| 2 | Malicious link/app | A link, website, downloaded file, application, or remote-access tool is the central pathway. |
| 3 | Account hacking | Unauthorized account access is described without a better-supported social-engineering or malicious-link pathway. |
| 4 | Impersonation/social engineering | Deception by a person or impersonated organization is central and is not better represented by codes 0 or 1. |
| 5 | Other/unknown | The pathway is absent, irreducibly ambiguous, or outside the preceding classes. |

When multiple pathways appear, annotate the pathway that most directly explains
the reported compromise according to the narrator. Use `Other/unknown` when a
choice would require speculation.

### 2. Attribution (`attribution`)

Single binary field describing where responsibility is expressed.

| Code | Label | Use when |
|---:|---|---|
| 0 | Internal attribution | The narrator expressly associates responsibility with their own action, omission, trust, mistake, or decision. |
| 1 | External attribution | Responsibility is associated with a perpetrator, organization, platform, technical event, or other outside actor. |

Annotate expressed attribution, not the annotator's judgment. If both appear,
follow the dominant explicit framing under the adjudication protocol.

### 3. Technical awareness (`awareness`)

Single ordinal field describing the specificity of the apparent mechanism in
the narrative.

| Code | Label | Definition |
|---:|---|---|
| 0 | No awareness | Primarily reports an outcome, such as a debit or loss of access, without an apparent causal pathway. |
| 1 | Partial awareness | Identifies a suspicious interaction or element but does not clearly connect it to the compromise mechanism. |
| 2 | Accurate awareness | Represents the apparent pathway with enough causal specificity to connect the relevant interaction or action to the outcome. |

Awareness measures expressed mechanistic detail, not cybersecurity expertise.
An otherwise accurate narrative may still contain a misconception.

### 4. Misconception (`misconception`)

Single binary field.

| Code | Label | Definition |
|---:|---|---|
| 0 | No identified misconception | The narrative does not contain a clearly supportable incorrect or materially incomplete causal/security belief. |
| 1 | Misconception present | At least one expressed explanation conflicts with the apparent security mechanism or materially omits a causal/control step. |

Apply conservatively. Missing detail alone is not necessarily a misconception.
The paper's non-exclusive descriptive subtypes are automatic-access belief,
false protective action, misattributed causality, and underestimated scope.

## Response dimensions

All response dimensions are multi-label. Set each indicator independently to
`1` only when explicitly supported; otherwise set it to `0`.

### 5. Emotional expression

| Field | Label | Evidence |
|---|---|---|
| `emotion_fear` | Fear | Fear, worry, threat, or apprehension. |
| `emotion_panic` | Panic | Acute urgency, panic, or overwhelming alarm. |
| `emotion_anger` | Anger | Anger, frustration, outrage, or resentment. |
| `emotion_confusion` | Confusion | Uncertainty about what happened, how, scope, consequences, or next steps. |
| `emotion_neutral` | Neutral/procedural | A substantially factual or procedural reporting tone. This may co-occur with another expressly stated emotion. |

Do not infer emotion solely from the seriousness of an incident.

### 6. Requested assistance

| Field | Label | Evidence |
|---|---|---|
| `assist_legal_action` | Legal action | Requests legal proceedings, punishment, or formal legal intervention. |
| `assist_fund_recovery` | Fund recovery | Requests return, reversal, freezing, or recovery of money. |
| `assist_account_recovery` | Account recovery | Requests restoration or securing of account access. |
| `assist_content_removal` | Content removal | Requests deletion, takedown, or removal of content/profile/material. |
| `assist_investigation` | Investigation | Requests tracing, identification, inquiry, or investigation. |

A reported past action is not automatically a request for assistance.

### 7. Post-incident action

| Field | Label | Evidence |
|---|---|---|
| `action_bank` | Bank informed | Reports contacting or notifying a bank/payment provider. |
| `action_password_reset` | Password reset | Reports changing/resetting credentials. |
| `action_police` | Police informed | Reports notifying police or another law-enforcement body. |
| `action_account_block` | Account blocked | Reports blocking, freezing, disabling, or securing an affected account/card where this is distinct from password reset. |

Intentions and requests are not completed actions. Annotate an action only when
the narrative states that it was already taken.

## Conservative decision rules

1. Ground every positive label in explicit narrative evidence.
2. Do not verify the complaint against outside sources.
3. Do not infer blame, intelligence, competence, intent, or emotional state.
4. Treat awareness and misconception independently.
5. Treat requests and completed actions independently.
6. Permit multiple emotion, assistance, and action labels.
7. Record uncertainty for adjudication rather than forcing an unsupported label.
8. Never place verbatim restricted narrative text in annotation notes.


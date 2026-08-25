"""Shared VIR field definitions for the public artifact."""

ATTACK_VECTOR = "attack_vector"
ATTRIBUTION = "attribution"
AWARENESS = "awareness"
MISCONCEPTION = "misconception"

EMOTIONS = [
    "emotion_panic",
    "emotion_anger",
    "emotion_fear",
    "emotion_confusion",
    "emotion_neutral",
]

ASSISTANCE = [
    "assist_fund_recovery",
    "assist_account_recovery",
    "assist_content_removal",
    "assist_legal_action",
    "assist_investigation",
]

ACTIONS = [
    "action_bank",
    "action_password_reset",
    "action_police",
    "action_account_block",
]

BINARY = [ATTRIBUTION, MISCONCEPTION]
MULTILABEL = EMOTIONS + ASSISTANCE + ACTIONS
LABEL_FIELDS = [ATTACK_VECTOR, ATTRIBUTION, AWARENESS, MISCONCEPTION] + MULTILABEL

ATTACK_VECTOR_NAMES = {
    0: "Phishing email",
    1: "Phishing SMS/call",
    2: "Malicious link/app",
    3: "Account hacking",
    4: "Impersonation/social engineering",
    5: "Other/unknown",
}

AWARENESS_NAMES = {0: "No awareness", 1: "Partial awareness", 2: "Accurate awareness"}

VALUE_RANGES = {
    ATTACK_VECTOR: set(range(6)),
    ATTRIBUTION: {0, 1},
    AWARENESS: {0, 1, 2},
    MISCONCEPTION: {0, 1},
    **{field: {0, 1} for field in MULTILABEL},
}

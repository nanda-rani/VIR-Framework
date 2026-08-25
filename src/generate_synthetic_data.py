#!/usr/bin/env python3
"""Generate invented VIR records for an end-to-end artifact demonstration.

The templates below were written for this artifact. They are not NCRP records,
not quotations, and not paraphrases of any individual complaint.
"""

import csv
from pathlib import Path

from vir_fields import LABEL_FIELDS


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "synthetic" / "vir_synthetic.csv"

TEMPLATES = [
    "I received an email that appeared to be from my bank. I opened the sign-in link and entered my details. Later I noticed an unauthorized transfer. Please investigate and help me recover the amount.",
    "A person called me and said that my bank account required verification. I shared the code received on my phone, after which money was debited. I informed my bank and request an investigation.",
    "I followed a support link and installed the application suggested there. Soon afterward I could not access my account. I changed my password and need help restoring the account.",
    "My online account is no longer accessible and I do not know how the access was changed. I have reset the password but remain worried about whether the account is secure.",
    "A person claiming to be a buyer contacted me and asked me to approve a payment request. The amount was deducted instead of credited. Please trace the person and recover the funds.",
    "I noticed a transaction that I did not authorize. I have not identified any message, call, link, or application connected with it. I contacted the bank and request clarification and recovery.",
    "I received a message saying that a delivery charge was pending. I used the attached payment link and later saw transactions that I did not recognize. I have blocked the account and need assistance.",
    "Someone introduced themselves as customer support and asked me to share a verification code. After I shared it, my account details were changed. I informed the police and request account recovery.",
    "I downloaded an application after being told it was required for a refund. The application asked for several permissions, and later money was transferred from my account. Please investigate the transfer.",
    "An unknown person gained access to my profile and posted content without my permission. I changed the password and request removal of the content and restoration of the account.",
    "A person used the name of a known service and asked me to send a payment to confirm my account. I made the payment and then realized the request was not genuine. I request legal action and fund recovery.",
    "An amount was deducted from my account without my approval. I am confused about how this happened and have asked the bank to block further transactions and examine the debit.",
]

LABEL_OVERRIDES = [
    {"attack_vector": 0, "attribution": 0, "awareness": 2, "emotion_neutral": 1, "assist_fund_recovery": 1, "assist_investigation": 1},
    {"attack_vector": 1, "attribution": 0, "awareness": 2, "emotion_neutral": 1, "assist_investigation": 1, "action_bank": 1},
    {"attack_vector": 2, "attribution": 1, "awareness": 2, "emotion_neutral": 1, "assist_account_recovery": 1, "action_password_reset": 1},
    {"attack_vector": 3, "attribution": 1, "awareness": 0, "emotion_fear": 1, "emotion_confusion": 1, "assist_account_recovery": 1, "action_password_reset": 1},
    {"attack_vector": 4, "attribution": 1, "awareness": 2, "emotion_neutral": 1, "assist_fund_recovery": 1, "assist_investigation": 1},
    {"attack_vector": 5, "attribution": 1, "awareness": 0, "emotion_confusion": 1, "assist_fund_recovery": 1, "assist_investigation": 1, "action_bank": 1},
    {"attack_vector": 2, "attribution": 1, "awareness": 2, "emotion_neutral": 1, "assist_fund_recovery": 1, "assist_account_recovery": 1, "action_account_block": 1},
    {"attack_vector": 4, "attribution": 0, "awareness": 2, "emotion_neutral": 1, "assist_account_recovery": 1, "action_police": 1},
    {"attack_vector": 2, "attribution": 1, "awareness": 2, "emotion_neutral": 1, "assist_fund_recovery": 1, "assist_investigation": 1},
    {"attack_vector": 3, "attribution": 1, "awareness": 0, "emotion_neutral": 1, "assist_account_recovery": 1, "assist_content_removal": 1, "action_password_reset": 1},
    {"attack_vector": 4, "attribution": 1, "awareness": 2, "emotion_neutral": 1, "assist_fund_recovery": 1, "assist_legal_action": 1},
    {"attack_vector": 5, "attribution": 1, "awareness": 0, "emotion_confusion": 1, "assist_fund_recovery": 1, "assist_investigation": 1, "action_bank": 1, "action_account_block": 1},
]


def labels_for(index: int) -> dict[str, int]:
    record = {
        "attack_vector": 5,
        "attribution": 1,
        "awareness": 0,
        "misconception": 0,
        "emotion_panic": 0,
        "emotion_anger": 0,
        "emotion_fear": 0,
        "emotion_confusion": 0,
        "emotion_neutral": 0,
        "assist_fund_recovery": 0,
        "assist_account_recovery": 0,
        "assist_content_removal": 0,
        "assist_legal_action": 0,
        "assist_investigation": 0,
        "action_bank": 0,
        "action_password_reset": 0,
        "action_police": 0,
        "action_account_block": 0,
    }
    record.update(LABEL_OVERRIDES[index % len(LABEL_OVERRIDES)])
    return record


def predicted(record: dict[str, int], index: int) -> dict[str, int]:
    out = {}
    for offset, field in enumerate(LABEL_FIELDS):
        value = record[field]
        # Deterministic, sparse errors make the evaluation demo non-trivial.
        if (index + offset * 7) % 29 == 0:
            if field == "attack_vector":
                value = (value + 1) % 6
            elif field == "awareness":
                value = min(2, value + 1) if value < 2 else 1
            else:
                value = 1 - value
        out[f"pred_{field}"] = value
    return out


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(60):
        labels = labels_for(index)
        row = {
            "record_id": f"SYN-{index + 1:04d}",
            "is_synthetic": 1,
            "split": ("train", "train", "validation", "test", "agreement")[index % 5],
            "narrative": TEMPLATES[index % len(TEMPLATES)],
            **labels,
            **predicted(labels, index),
        }
        rows.append(row)

    fields = list(rows[0])
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} wholly synthetic records to {OUTPUT}")


if __name__ == "__main__":
    main()

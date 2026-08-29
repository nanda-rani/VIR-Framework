# Data Availability and Non-Redistribution Statement

## Provenance

The study analyzes complaints submitted to India's National Cybercrime
Reporting Portal (NCRP), operated by the Indian Cybercrime Coordination Centre
(I4C). The data were made available through participation in the CyberGuard AI
Hackathon 2024, organized by IndiaAI and I4C. The authors received an
anonymized version and did not receive direct identifiers such as names,
contact details, or account identifiers.

## Why complaint-level data are not shared

The authors are not the owners or controllers of the NCRP data and do not have
authorization to redistribute them. The narratives concern sensitive victim
experiences. Free text can contain indirect identifiers, distinctive event
details, financial information, or rare combinations of facts after direct
identifiers have been removed. Public release could therefore create privacy,
re-identification, and misuse risks.

| Unavailable artifact | Reason | Public substitute |
|---|---|---|
| Raw complaint corpus | No redistribution authority; sensitive victim narratives | Provenance, corpus-level counts, schema, and synthetic examples |
| Human-annotated records | Labels remain linkable to sensitive text and row structure | Full VIR codebook and synthetic labelled records |
| Automatically labelled records | Complaint-level derivatives retain linkability and sensitive categories | Aggregate results and aggregate-analysis code |
| Train/validation/test memberships and row identifiers | May facilitate linkage or reconstruction | 70/15/15 split procedure, privacy-preserving split fingerprints, and synthetic splits |
| Embeddings, token caches, and raw model outputs | May encode or reproduce complaint text | Architecture and configuration documentation |
| Fine-tuned model weights | Memorization of sensitive narratives has not been ruled out | Training, testing, evaluation, and inference source plus complete configuration |

## Access

The artifact does not offer a controlled-access copy. Researchers must obtain
authorization independently from the relevant data owner or controller. The
authors cannot guarantee access and will not transfer records obtained under
their own access arrangement.


Nothing in this artifact grants permission to identify, contact, trace, or
profile complainants or to combine materials with external data for
re-identification.

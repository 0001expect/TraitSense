"""
40 Likert-scale questionnaire items for rule-based MBTI scoring.

Completely separate from the TF-IDF / ML text pipeline.
Each item maps to one dimension and the letter that "Agree" supports.
"""

from __future__ import annotations

from typing import Dict, List, TypedDict


class Question(TypedDict):
    id: int
    section: str
    axis: str  # IE | NS | TF | JP
    towards: str  # letter that Strongly Agree pushes toward
    text: str


SECTIONS = [
    "Social Behaviour",
    "Information Processing",
    "Decision Making",
    "Lifestyle & Planning",
]

LIKERT_OPTIONS = [
    {"value": 5, "label": "Strongly Agree"},
    {"value": 4, "label": "Agree"},
    {"value": 3, "label": "Neutral"},
    {"value": 2, "label": "Disagree"},
    {"value": 1, "label": "Strongly Disagree"},
]

QUESTIONNAIRE_ITEMS: List[Question] = [
    # ---- Social Behaviour (IE) ----
    {
        "id": 1,
        "section": "Social Behaviour",
        "axis": "IE",
        "towards": "E",
        "text": "I feel energized after spending time with a group of people.",
    },
    {
        "id": 2,
        "section": "Social Behaviour",
        "axis": "IE",
        "towards": "I",
        "text": "I prefer spending quiet evenings alone or with one close person rather than going out.",
    },
    {
        "id": 3,
        "section": "Social Behaviour",
        "axis": "IE",
        "towards": "E",
        "text": "I enjoy meeting new people and starting conversations easily.",
    },
    {
        "id": 4,
        "section": "Social Behaviour",
        "axis": "IE",
        "towards": "I",
        "text": "I need alone time to recharge after busy social events.",
    },
    {
        "id": 5,
        "section": "Social Behaviour",
        "axis": "IE",
        "towards": "E",
        "text": "I think out loud and process ideas through discussion.",
    },
    {
        "id": 6,
        "section": "Social Behaviour",
        "axis": "IE",
        "towards": "I",
        "text": "I usually think things through carefully before sharing them with others.",
    },
    {
        "id": 7,
        "section": "Social Behaviour",
        "axis": "IE",
        "towards": "E",
        "text": "Being around lively environments makes me feel more productive.",
    },
    {
        "id": 8,
        "section": "Social Behaviour",
        "axis": "IE",
        "towards": "I",
        "text": "I would rather write a message than speak on the phone when possible.",
    },
    {
        "id": 9,
        "section": "Social Behaviour",
        "axis": "IE",
        "towards": "E",
        "text": "I often take the lead in starting group activities or conversations.",
    },
    {
        "id": 10,
        "section": "Social Behaviour",
        "axis": "IE",
        "towards": "I",
        "text": "I am more comfortable observing first before joining a social setting.",
    },
    # ---- Information Processing (NS) ----
    {
        "id": 11,
        "section": "Information Processing",
        "axis": "NS",
        "towards": "S",
        "text": "I trust facts, details, and real-world experience more than theories.",
    },
    {
        "id": 12,
        "section": "Information Processing",
        "axis": "NS",
        "towards": "N",
        "text": "I enjoy imagining future possibilities and abstract ideas.",
    },
    {
        "id": 13,
        "section": "Information Processing",
        "axis": "NS",
        "towards": "S",
        "text": "I prefer step-by-step instructions with clear, practical examples.",
    },
    {
        "id": 14,
        "section": "Information Processing",
        "axis": "NS",
        "towards": "N",
        "text": "I often notice patterns and connections that others overlook.",
    },
    {
        "id": 15,
        "section": "Information Processing",
        "axis": "NS",
        "towards": "S",
        "text": "I focus on what is happening right now rather than distant hypotheticals.",
    },
    {
        "id": 16,
        "section": "Information Processing",
        "axis": "NS",
        "towards": "N",
        "text": "I get excited by new concepts even if they are not immediately useful.",
    },
    {
        "id": 17,
        "section": "Information Processing",
        "axis": "NS",
        "towards": "S",
        "text": "I am good at remembering specific details about past events.",
    },
    {
        "id": 18,
        "section": "Information Processing",
        "axis": "NS",
        "towards": "N",
        "text": "I often think about 'what could be' instead of only 'what is'.",
    },
    {
        "id": 19,
        "section": "Information Processing",
        "axis": "NS",
        "towards": "S",
        "text": "I prefer proven methods over experimental, untested approaches.",
    },
    {
        "id": 20,
        "section": "Information Processing",
        "axis": "NS",
        "towards": "N",
        "text": "Metaphors, symbolism, and big-picture stories resonate with me.",
    },
    # ---- Decision Making (TF) ----
    {
        "id": 21,
        "section": "Decision Making",
        "axis": "TF",
        "towards": "T",
        "text": "I prioritize logic and objective criteria when making decisions.",
    },
    {
        "id": 22,
        "section": "Decision Making",
        "axis": "TF",
        "towards": "F",
        "text": "I consider how people will feel before I decide what to do.",
    },
    {
        "id": 23,
        "section": "Decision Making",
        "axis": "TF",
        "towards": "T",
        "text": "I am comfortable giving honest criticism if it improves the outcome.",
    },
    {
        "id": 24,
        "section": "Decision Making",
        "axis": "TF",
        "towards": "F",
        "text": "Maintaining harmony in relationships is often more important than being right.",
    },
    {
        "id": 25,
        "section": "Decision Making",
        "axis": "TF",
        "towards": "T",
        "text": "I evaluate arguments based on consistency and evidence, not personal preference.",
    },
    {
        "id": 26,
        "section": "Decision Making",
        "axis": "TF",
        "towards": "F",
        "text": "I naturally empathize with others' emotions during conflicts.",
    },
    {
        "id": 27,
        "section": "Decision Making",
        "axis": "TF",
        "towards": "T",
        "text": "Fairness means applying the same rules to everyone equally.",
    },
    {
        "id": 28,
        "section": "Decision Making",
        "axis": "TF",
        "towards": "F",
        "text": "Fairness means considering each person's unique situation and needs.",
    },
    {
        "id": 29,
        "section": "Decision Making",
        "axis": "TF",
        "towards": "T",
        "text": "I stay calm and analytical even when others become emotional.",
    },
    {
        "id": 30,
        "section": "Decision Making",
        "axis": "TF",
        "towards": "F",
        "text": "Personal values and compassion guide my hardest choices.",
    },
    # ---- Lifestyle & Planning (JP) ----
    {
        "id": 31,
        "section": "Lifestyle & Planning",
        "axis": "JP",
        "towards": "J",
        "text": "I like having a clear plan and schedule for my week.",
    },
    {
        "id": 32,
        "section": "Lifestyle & Planning",
        "axis": "JP",
        "towards": "P",
        "text": "I prefer keeping my options open rather than locking into plans early.",
    },
    {
        "id": 33,
        "section": "Lifestyle & Planning",
        "axis": "JP",
        "towards": "J",
        "text": "Completing tasks before the deadline reduces my stress.",
    },
    {
        "id": 34,
        "section": "Lifestyle & Planning",
        "axis": "JP",
        "towards": "P",
        "text": "I do my best work when I can adapt spontaneously to changing situations.",
    },
    {
        "id": 35,
        "section": "Lifestyle & Planning",
        "axis": "JP",
        "towards": "J",
        "text": "I enjoy organizing spaces, files, and routines so everything has a place.",
    },
    {
        "id": 36,
        "section": "Lifestyle & Planning",
        "axis": "JP",
        "towards": "P",
        "text": "Too much structure makes me feel restricted.",
    },
    {
        "id": 37,
        "section": "Lifestyle & Planning",
        "axis": "JP",
        "towards": "J",
        "text": "I feel satisfied when decisions are finalized and settled.",
    },
    {
        "id": 38,
        "section": "Lifestyle & Planning",
        "axis": "JP",
        "towards": "P",
        "text": "I often wait for more information before committing to a decision.",
    },
    {
        "id": 39,
        "section": "Lifestyle & Planning",
        "axis": "JP",
        "towards": "J",
        "text": "I prefer lists, goals, and measurable progress markers.",
    },
    {
        "id": 40,
        "section": "Lifestyle & Planning",
        "axis": "JP",
        "towards": "P",
        "text": "I enjoy exploring new ideas even if they disrupt my original plan.",
    },
]


def questions_payload() -> Dict:
    """JSON-serializable payload for the frontend."""
    return {
        "sections": SECTIONS,
        "likert": LIKERT_OPTIONS,
        "questions": QUESTIONNAIRE_ITEMS,
        "total": len(QUESTIONNAIRE_ITEMS),
    }

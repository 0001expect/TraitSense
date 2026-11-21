"""MBTI type profiles shown in the Flask UI (descriptions, careers, styles)."""

from __future__ import annotations

from typing import Dict

MBTI_PROFILES: Dict[str, Dict[str, object]] = {
    "INTJ": {
        "title": "Architect",
        "description": "Strategic, independent, and long-range thinkers who value competence.",
        "strengths": ["Strategic planning", "Independence", "High standards", "Analytical depth"],
        "weaknesses": ["Can seem aloof", "Overly critical", "Impatient with inefficiency"],
        "careers": ["Software architect", "Research scientist", "Strategy consultant", "Systems engineer"],
        "learning_style": "Self-directed learning with frameworks, models, and deep solitary study.",
        "communication_style": "Direct, precise, and idea-focused; prefers written clarity over small talk.",
    },
    "INTP": {
        "title": "Logician",
        "description": "Curious analysts who love abstract systems and intellectual exploration.",
        "strengths": ["Logical reasoning", "Creativity", "Objectivity", "Open-mindedness"],
        "weaknesses": ["Procrastination", "Overlooks practical details", "Socially reserved"],
        "careers": ["Data scientist", "Philosopher", "Software developer", "Academic researcher"],
        "learning_style": "Explores concepts through experimentation, debate, and curiosity-driven rabbit holes.",
        "communication_style": "Thoughtful and abstract; may jump between theories before conclusions.",
    },
    "ENTJ": {
        "title": "Commander",
        "description": "Decisive organizers who enjoy leading projects and driving results.",
        "strengths": ["Leadership", "Decisiveness", "Efficiency", "Confidence"],
        "weaknesses": ["Can be domineering", "Impatient", "Overlooks emotions"],
        "careers": ["Executive", "Entrepreneur", "Operations manager", "Lawyer"],
        "learning_style": "Goal-oriented practice, mentorship, and competition that measures progress.",
        "communication_style": "Assertive and structured; focuses on outcomes and accountability.",
    },
    "ENTP": {
        "title": "Debater",
        "description": "Inventive conversationalists who thrive on ideas, novelty, and challenge.",
        "strengths": ["Quick wit", "Innovation", "Adaptability", "Persuasion"],
        "weaknesses": ["Restless", "Argumentative", "Follow-through gaps"],
        "careers": ["Product manager", "Startup founder", "Marketing strategist", "Journalist"],
        "learning_style": "Learns by debating, prototyping, and connecting disparate ideas.",
        "communication_style": "Energetic, playful, and provocative; loves exploring alternatives aloud.",
    },
    "INFJ": {
        "title": "Advocate",
        "description": "Insightful idealists guided by values and a desire to help others grow.",
        "strengths": ["Empathy", "Vision", "Integrity", "Insight into people"],
        "weaknesses": ["Perfectionism", "Burnout risk", "Hard on themselves"],
        "careers": ["Counselor", "Writer", "HR specialist", "Nonprofit leader"],
        "learning_style": "Meaning-driven study with reflection, mentoring, and purpose-linked goals.",
        "communication_style": "Warm but private; prefers deep one-to-one conversations.",
    },
    "INFP": {
        "title": "Mediator",
        "description": "Creative, authentic individuals guided by personal values and imagination.",
        "strengths": ["Creativity", "Empathy", "Authenticity", "Idealism"],
        "weaknesses": ["Overly idealistic", "Conflict-averse", "Self-critical"],
        "careers": ["Designer", "Therapist", "Author", "UX researcher"],
        "learning_style": "Explores topics connected to personal meaning, stories, and creative projects.",
        "communication_style": "Gentle and reflective; expresses feelings through nuance and metaphor.",
    },
    "ENFJ": {
        "title": "Protagonist",
        "description": "Charismatic mentors who inspire and organize people around shared goals.",
        "strengths": ["Empathy", "Leadership", "Communication", "Organization"],
        "weaknesses": ["People-pleasing", "Overcommitted", "Sensitive to criticism"],
        "careers": ["Teacher", "Coach", "Public relations", "Community organizer"],
        "learning_style": "Collaborative learning, teaching others, and applied group projects.",
        "communication_style": "Warm, persuasive, and encouraging; attuned to emotional climate.",
    },
    "ENFP": {
        "title": "Campaigner",
        "description": "Enthusiastic explorers energized by people, possibilities, and creative freedom.",
        "strengths": ["Enthusiasm", "Creativity", "People skills", "Optimism"],
        "weaknesses": ["Distractible", "Overcommits", "Dislikes routine"],
        "careers": ["Creative director", "Counselor", "Entrepreneur", "Event producer"],
        "learning_style": "Interactive brainstorming, storytelling, and flexible project-based learning.",
        "communication_style": "Expressive and spontaneous; connects ideas to people and possibilities.",
    },
    "ISTJ": {
        "title": "Logistician",
        "description": "Reliable realists who value duty, accuracy, and proven methods.",
        "strengths": ["Dependability", "Organization", "Attention to detail", "Integrity"],
        "weaknesses": ["Resistance to change", "Overly rigid", "Reserved"],
        "careers": ["Accountant", "Auditor", "Project coordinator", "Military / operations roles"],
        "learning_style": "Step-by-step materials, checklists, and practice with clear standards.",
        "communication_style": "Factual and concise; prefers agendas and concrete evidence.",
    },
    "ISFJ": {
        "title": "Defender",
        "description": "Warm guardians who protect traditions, people, and practical harmony.",
        "strengths": ["Loyalty", "Practical support", "Patience", "Observation"],
        "weaknesses": ["Self-sacrificing", "Avoids conflict", "Reluctant to change"],
        "careers": ["Nurse", "Administrator", "Librarian", "Customer success"],
        "learning_style": "Guided practice, examples, and supportive feedback in steady routines.",
        "communication_style": "Considerate and concrete; listens carefully before responding.",
    },
    "ESTJ": {
        "title": "Executive",
        "description": "Organizers who bring order, standards, and accountable execution.",
        "strengths": ["Organization", "Decisiveness", "Reliability", "Leadership"],
        "weaknesses": ["Inflexible", "Blunt", "Can dismiss feelings"],
        "careers": ["Operations manager", "Judge", "School administrator", "Supply chain lead"],
        "learning_style": "Structured curricula, deadlines, and measurable skill drills.",
        "communication_style": "Clear directives and expectations; values punctual follow-through.",
    },
    "ESFJ": {
        "title": "Consul",
        "description": "Sociable caregivers who strengthen communities and relationships.",
        "strengths": ["Warmth", "Cooperation", "Practical help", "Social awareness"],
        "weaknesses": ["Needs approval", "Conflict-sensitive", "Overextends socially"],
        "careers": ["Healthcare coordinator", "Event planner", "Teacher", "Social worker"],
        "learning_style": "Group study, mentoring, and applied people-centered practice.",
        "communication_style": "Friendly and affirming; maintains group harmony.",
    },
    "ISTP": {
        "title": "Virtuoso",
        "description": "Hands-on problem solvers who explore how things work through action.",
        "strengths": ["Practical troubleshooting", "Calm under pressure", "Adaptability"],
        "weaknesses": ["Reserved emotions", "Easily bored", "Commitment hesitation"],
        "careers": ["Mechanic", "Engineer", "Pilot", "Forensic analyst"],
        "learning_style": "Learning by doing—labs, tools, and real-world debugging.",
        "communication_style": "Spare and concrete; shows competence through action more than words.",
    },
    "ISFP": {
        "title": "Adventurer",
        "description": "Gentle aesthetics-driven people living through sensory experience and values.",
        "strengths": ["Artistic sensibility", "Flexibility", "Kindness", "Present focus"],
        "weaknesses": ["Avoids long plans", "Conflict-averse", "Underestimates self"],
        "careers": ["Artist", "Chef", "Designer", "Veterinarian"],
        "learning_style": "Sensory, creative projects with freedom and low-pressure guidance.",
        "communication_style": "Soft-spoken and sincere; prefers showing care through deeds.",
    },
    "ESTP": {
        "title": "Entrepreneur",
        "description": "Action-oriented realists who thrive on momentum, risk, and immediate results.",
        "strengths": ["Boldness", "Persuasion", "Situational awareness", "Energy"],
        "weaknesses": ["Impulsive", "Easily bored", "Overlooks long-term costs"],
        "careers": ["Sales", "Emergency responder", "Sports coach", "Negotiator"],
        "learning_style": "Fast feedback loops, competitions, and live practice scenarios.",
        "communication_style": "Candid, energetic, and in-the-moment; keeps conversations moving.",
    },
    "ESFP": {
        "title": "Entertainer",
        "description": "Spontaneous, warm performers who energize groups and savor the present.",
        "strengths": ["Charisma", "Optimism", "Practicality", "People skills"],
        "weaknesses": ["Distractible", "Conflict-avoidant", "Dislikes heavy theory"],
        "careers": ["Performer", "Hospitality", "Tour guide", "Brand ambassador"],
        "learning_style": "Interactive workshops, storytelling, and hands-on group activities.",
        "communication_style": "Lively and expressive; connects through humor and shared experiences.",
    },
}

# Backward-compatible short map used by older imports
MBTI_DESCRIPTIONS = {
    code: f"{profile['title']} — {profile['description']}"
    for code, profile in MBTI_PROFILES.items()
}

DISCLAIMER = (
    "This prediction is based on language patterns learned from training data. "
    "It is an estimate and should not be treated as a definitive psychological assessment."
)

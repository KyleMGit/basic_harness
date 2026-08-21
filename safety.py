"""Small deterministic host-side screening for persisted prompt material."""
import re


INJECTION_PATTERNS = (
    r"\bignore (?:all |any )?(?:previous|prior|above) instructions\b",
    r"\boverride (?:the )?(?:system|developer|prompt) instructions\b",
    r"\b(?:must|always) call (?:the )?[a-z_]+ tool\b",
    r"<\/?(?:system|developer|tool_call)>",
    r"\bdisregard your (?:instructions|rules)\b",
    r"\bforget your instructions\b",
    r"\bnew instructions\s*:",
    r"\]\]>",
)

SENSITIVE_COMMAND = re.compile(
    r"\b(?:reveal|exfiltrate|print|send) (?:all )?(?:secrets?|credentials?|api keys?|tokens?)\b",
    re.IGNORECASE,
)
NEGATION = re.compile(r"(?:\bdo\s+not\b|\bdon't\b|\bnever\b)\s+$", re.IGNORECASE)
ROLE_REASSIGNMENT = re.compile(
    r"\byou are now\s+(?:(?:an?|the)\s+)?(?:root|administrator|admin|system|developer|superuser|unrestricted)\b",
    re.IGNORECASE,
)


def screen_prompt_content(content: str) -> tuple[bool, str]:
    text = content or ""
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return False, "Rejected/quarantined: content resembles prompt injection or tool-forcing instructions."
    if ROLE_REASSIGNMENT.search(text):
        return False, "Rejected/quarantined: content resembles prompt injection or tool-forcing instructions."
    if re.search(r"(?im)^\s*(?:[-*]\s*)?system prompt\s*:", text):
        return False, "Rejected/quarantined: content resembles prompt injection or tool-forcing instructions."
    for match in SENSITIVE_COMMAND.finditer(text):
        clause_prefix = re.split(r"[.;:!?\n]", text[:match.start()])[-1]
        if not NEGATION.search(clause_prefix):
            return False, "Rejected/quarantined: content resembles prompt injection or tool-forcing instructions."
    return True, "accepted"

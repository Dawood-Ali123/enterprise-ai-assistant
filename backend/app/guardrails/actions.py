import re
from typing import Optional
from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome

@action(is_system_action=True)
async def check_enterprise_input(
    context:Optional[dict]=None
):
    user_message=""
    if context:
        user_message=(
            context.get("user_message")
            or context.get("last_user_message")
            or ""
        )
    text=user_message.lower().strip()
    injection_patterns=[
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(all\s+)?instructions",
        r"forget\s+(your\s+)?instructions",
        r"forget\s+(your\s+)?system\s+prompt",
        r"ignore\s+your\s+system\s+prompt",
        r"override\s+(your\s+)?instructions",
        r"bypass\s+(your\s+)?rules",
        r"disable\s+(your\s+)?safety",
        r"you\s+are\s+now\s+(dan|jailbroken|unrestricted)",
        r"act\s+as\s+an\s+unrestricted\s+ai",
        r"reveal\s+(your\s+)?system\s+prompt",
        r"show\s+(me\s+)?your\s+system\s+prompt",
        r"print\s+(your\s+)?system\s+prompt",
    ]
    for pattern in injection_patterns:
        if re.search(pattern,text):
            return RailOutcome.block(
                reason="Potential Prompt injection or jailbreak detected",
                metadata={
                    "category":"prompt_injection_or_jailbreak"
                }
            )
    off_topic_patterns=[
        r"\brecipe\b",
        r"\bcook(ing)?\b",
        r"\bfootball\b",
        r"\bcricket\b",
        r"\bmovie\b",
        r"\bsong\b",
        r"\bmusic\b",
        r"\bvideo game\b",
        r"\bgaming\b",
        r"\bweather\b",
        r"\btravel\b",
        r"\bhotel\b",
        r"\bdating\b",
        r"\bhoroscope\b",
    ]
    for pattern in off_topic_patterns:
        if re.search(pattern,text):
            return RailOutcome.block(
                reason="Request is outside the enterprise assistant scope",
                metadata={
                    "category":"Off_topic"
                }
            )
    return RailOutcome.allow(
        reason="Input passed the enterprise security checks",
        metadata={
            "category":"allowed"
        }
    )
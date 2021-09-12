from enum import Enum


class SkillShape(str, Enum):
    UNKNOWN = "UNKNOWN"
    STRAIGHT = "STRAIGHT"
    TUCK = "TUCK"
    PIKE = "PIKE"
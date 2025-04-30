from core.models import UserProfile
from core.serializers import UserProfileForLLM
import structlog

log = structlog.get_logger(__name__)


def load_profile(name: str = "Test User 2 - Full Data Load") -> UserProfileForLLM:
    user_profile = UserProfile.objects.get(name=name)
    log.info("loaded_user_profile", user_name=user_profile.name)

    llm_user_profile = UserProfileForLLM(user_profile).data

    return {
        "name": name,
        "llm_user_profile": llm_user_profile
    }
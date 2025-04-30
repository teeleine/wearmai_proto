from django.core.management.base import BaseCommand
from services.llm_coach.coach_service import CoachService
from user_profile.loader import load_profile
from infrastructure.llm_clients.base import LLModels
import structlog

log = structlog.get_logger(__name__)

debug = False

class Command(BaseCommand):
    help = "Runs LLM-based Coach"

    def add_arguments(self, parser):
        parser.add_argument(
            "--debug", 
            action="store_true", 
            help="Enable debug mode"
        )

    def handle(self, *args, **options):
        self.debug = options.get('debug', False)
        if(self.debug):
            log.info("coach_cli_debug_mode")

        user_profile = load_profile(name="Test User 2 - Full Data Load")

        coach_svc = CoachService("BookChunks_voyage",user_profile['llm_user_profile'])
        coach_response = coach_svc.send_question(query="I am planning to join the Amsterdam marathon in 4 months. Could you generate my personal training plan?",model=LLModels.GEMINI_25_FLASH,temperature=0.7,thinking_budget=0)
        log.info("received_coach_response", coach_response=coach_response)
        coach_svc.vectorstore.close()

    

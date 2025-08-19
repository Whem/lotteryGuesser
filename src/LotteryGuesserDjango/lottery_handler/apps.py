from django.apps import AppConfig


class LotteryhandlerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'lottery_handler'
    
    def ready(self):
        """Alkalmazás inicializálása"""
        try:
            # Logging és figyelmeztetések beállítása
            from .utils import setup_logging, suppress_warnings
            
            # Figyelmeztetések elnémítása
            suppress_warnings()
            
            # Logging inicializálása
            logger = setup_logging()
            logger.info("[INIT] Lottery Handler alkalmazas inicializalva")
            logger.info("[INIT] TensorFlow es MLxtend figyelmeztetesek elnemitva")
            
        except Exception as e:
            print(f"Hiba az alkalmazás inicializálásában: {e}")

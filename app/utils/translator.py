import json
from functools import reduce
from pathlib import Path
from typing import Optional, Any, Callable

from loguru import logger
from pydantic import ValidationError, BaseModel

from app.config import config

TRANSLATIONS_PATH = Path(f"{config.translation_path}/{config.language_type}.json")


class TranslationModel(BaseModel):
    TRANSLATION_BY: str
    TRANSLATION_VERSION: str
    DATABASE: dict[str, dict[str, str]]
    RESOURCE: dict[str, dict[str, str]]
    MODEL: dict[str, dict[str, str]]
    ERRORS: dict[str, Any]


class Translator:
    def __init__(self) -> None:
        self.translations: Optional[TranslationModel] = None
        try:
            with open(TRANSLATIONS_PATH, encoding="utf-8") as f:
                translations_data = json.load(f)
            self.translations = TranslationModel(**translations_data)
        except FileNotFoundError:
            logger.error(f"Translation file {TRANSLATIONS_PATH} not found.")
        except ValidationError as e:
            logger.error(f"Translation file format error: {e}")

    def _extract_value(self, key_sequence: str, default: Optional[str] = None) -> Optional[str]:
        keys = key_sequence.split(".")
        try:
            return reduce(
                lambda data, key: data.get(key, default) if isinstance(data, dict) else default,
                keys,
                self.translations.model_dump(),
            )
        except Exception as e:
            logger.warning(f"Error extracting translation for key sequence '{key_sequence}': {e}")
            return default

    def get_local_translations(self) -> Callable[[str, Optional[str]], Optional[str]]:
        def translator(message_key: str, default: Optional[str] = None) -> Optional[str]:
            return self._extract_value(message_key, default)

        return translator


message_translator = Translator().get_local_translations()

"""Pure character-sequence preparation for the frozen RQ1 transformer."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .phiusiil import PreparationError, canonicalize_url

MAX_SEQUENCE_LENGTH = 256
HEAD_LENGTH = 192
TAIL_LENGTH = 64
PAD_ID = 0
UNK_ID = 1
CHARACTER_IDS_START = 2
NORMALIZATION_ID = "canonical-url-v1+utf8-percent-ascii-v1"

_VOCABULARY_SCHEMA_KEYS = {
    "character_ids_start",
    "characters",
    "max_sequence_length",
    "normalization",
    "pad_id",
    "schema_version",
    "unk_id",
}


class CharacterSequenceError(ValueError):
    """Raised when a URL or vocabulary violates the frozen sequence contract."""


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CharacterSequenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def truncate_character_sequence(sequence: object) -> str:
    """Apply the frozen first-192, last-64 retention rule."""
    if type(sequence) is not str or not sequence or not sequence.isascii():
        raise CharacterSequenceError("sequence must be a nonempty ASCII string")
    if len(sequence) <= MAX_SEQUENCE_LENGTH:
        return sequence
    return sequence[:HEAD_LENGTH] + sequence[-TAIL_LENGTH:]


def normalize_character_url(raw_url: object) -> str:
    """Canonicalize a raw URL and return its retained ASCII representation."""
    try:
        canonical_url = canonicalize_url(raw_url)
    except PreparationError:
        raise CharacterSequenceError(
            "raw_url is missing or invalid under canonical-url-v1"
        ) from None

    try:
        ascii_url = "".join(
            character
            if character.isascii()
            else "".join(f"%{byte:02X}" for byte in character.encode("utf-8"))
            for character in canonical_url
        )
    except UnicodeEncodeError:
        raise CharacterSequenceError(
            "raw_url is missing or invalid under canonical-url-v1"
        ) from None
    return truncate_character_sequence(ascii_url)


@dataclass(frozen=True)
class CharacterVocabulary:
    """Immutable, ASCII-sorted character vocabulary learned from training URLs."""

    characters: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.characters) is not tuple:
            raise CharacterSequenceError("characters must be an exact tuple")
        if not self.characters:
            raise CharacterSequenceError("characters must not be empty")
        if any(
            type(character) is not str or len(character) != 1 or not character.isascii()
            for character in self.characters
        ):
            raise CharacterSequenceError("characters must be single ASCII characters")
        if self.characters != tuple(sorted(set(self.characters), key=ord)):
            raise CharacterSequenceError("characters must be unique and ASCII-sorted")

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    @property
    def character_to_id(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                character: index
                for index, character in enumerate(
                    self.characters, start=CHARACTER_IDS_START
                )
            }
        )

    @property
    def size(self) -> int:
        return len(self.characters) + CHARACTER_IDS_START

    def to_json(self) -> str:
        """Serialize the vocabulary using one deterministic JSON representation."""
        payload = {
            "character_ids_start": CHARACTER_IDS_START,
            "characters": list(self.characters),
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "normalization": NORMALIZATION_ID,
            "pad_id": PAD_ID,
            "schema_version": 1,
            "unk_id": UNK_ID,
        }
        return (
            json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            )
            + "\n"
        )

    @classmethod
    def from_json(cls, serialized: object) -> "CharacterVocabulary":
        """Load a vocabulary only when every frozen schema field matches."""
        if type(serialized) is not str:
            raise CharacterSequenceError(
                "serialized vocabulary must be an exact string"
            )
        try:
            payload = json.loads(
                serialized, object_pairs_hook=_object_without_duplicate_keys
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise CharacterSequenceError(
                "serialized vocabulary is not valid JSON"
            ) from None
        if type(payload) is not dict or set(payload) != _VOCABULARY_SCHEMA_KEYS:
            raise CharacterSequenceError("serialized vocabulary schema is invalid")
        expected_constants = {
            "character_ids_start": CHARACTER_IDS_START,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "normalization": NORMALIZATION_ID,
            "pad_id": PAD_ID,
            "schema_version": 1,
            "unk_id": UNK_ID,
        }
        if any(
            type(payload[key]) is not type(value) or payload[key] != value
            for key, value in expected_constants.items()
        ):
            raise CharacterSequenceError("serialized vocabulary constants are invalid")
        if type(payload["characters"]) is not list:
            raise CharacterSequenceError("serialized characters must be a list")
        return cls(tuple(payload["characters"]))


@dataclass(frozen=True)
class EncodedCharacterSequence:
    """Fixed-width token IDs and a padding mask for one URL."""

    token_ids: tuple[int, ...]
    padding_mask: tuple[bool, ...]
    length: int

    def __post_init__(self) -> None:
        if (
            type(self.token_ids) is not tuple
            or len(self.token_ids) != MAX_SEQUENCE_LENGTH
            or any(
                type(token_id) is not int or token_id < PAD_ID
                for token_id in self.token_ids
            )
        ):
            raise CharacterSequenceError(
                "token_ids must contain 256 nonnegative integers"
            )
        if (
            type(self.padding_mask) is not tuple
            or len(self.padding_mask) != MAX_SEQUENCE_LENGTH
            or any(type(value) is not bool for value in self.padding_mask)
        ):
            raise CharacterSequenceError("padding_mask must contain 256 booleans")
        if type(self.length) is not int or not 1 <= self.length <= MAX_SEQUENCE_LENGTH:
            raise CharacterSequenceError("length must be between 1 and 256")
        expected_mask = (False,) * self.length + (True,) * (
            MAX_SEQUENCE_LENGTH - self.length
        )
        if self.padding_mask != expected_mask:
            raise CharacterSequenceError("padding_mask does not match sequence length")
        if any(token_id == PAD_ID for token_id in self.token_ids[: self.length]) or any(
            token_id != PAD_ID for token_id in self.token_ids[self.length :]
        ):
            raise CharacterSequenceError("token_ids do not use right padding")


def build_character_vocabulary(training_urls: object) -> CharacterVocabulary:
    """Build an ASCII-sorted vocabulary from normalized training URLs only."""
    if isinstance(training_urls, (str, bytes)) or not isinstance(
        training_urls, Iterable
    ):
        raise CharacterSequenceError("training_urls must be an iterable of raw URLs")

    observed: set[str] = set()
    count = 0
    for raw_url in training_urls:
        count += 1
        observed.update(normalize_character_url(raw_url))
    if count == 0:
        raise CharacterSequenceError("training_urls must not be empty")
    return CharacterVocabulary(tuple(sorted(observed, key=ord)))


def encode_character_url(
    raw_url: object, vocabulary: CharacterVocabulary
) -> EncodedCharacterSequence:
    """Normalize and encode one URL with fixed right padding."""
    if type(vocabulary) is not CharacterVocabulary:
        raise CharacterSequenceError("vocabulary must be a CharacterVocabulary")

    sequence = normalize_character_url(raw_url)
    character_to_id = vocabulary.character_to_id
    retained_ids = tuple(
        character_to_id.get(character, UNK_ID) for character in sequence
    )
    length = len(retained_ids)
    padding_count = MAX_SEQUENCE_LENGTH - length
    return EncodedCharacterSequence(
        token_ids=retained_ids + (PAD_ID,) * padding_count,
        padding_mask=(False,) * length + (True,) * padding_count,
        length=length,
    )

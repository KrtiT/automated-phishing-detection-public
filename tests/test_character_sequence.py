import json

import pytest

from automated_phishing_detection import character_sequence


def test_normalization_reuses_canonical_url_rules_and_returns_ascii():
    normalized = character_sequence.normalize_character_url(
        "HTTPS://BÜCHER.example:443/a/β?q=%2f&next=/x#é"
    )

    assert normalized == ("https://xn--bcher-kva.example/a/%CE%B2?q=%2F&next=/x#%C3%A9")
    assert normalized.isascii()


def test_normalization_preserves_existing_escapes_and_reserved_characters():
    normalized = character_sequence.normalize_character_url(
        "https://example.test/a%2fb;v=1?q=x/y?z:@&k=$+,!#part"
    )

    assert normalized == "https://example.test/a%2Fb;v=1?q=x/y?z:@&k=$+,!#part"


def test_truncation_retains_first_192_and_last_64_characters():
    sequence = "a" * 192 + "discarded" + "z" * 64

    assert character_sequence.truncate_character_sequence(sequence) == (
        "a" * 192 + "z" * 64
    )
    assert character_sequence.truncate_character_sequence("x" * 256) == "x" * 256


def test_vocabulary_is_train_only_sorted_and_reserves_pad_and_unk_ids():
    vocabulary = character_sequence.build_character_vocabulary(
        ["https://z.example/B", "https://a.example/a"]
    )

    assert vocabulary.pad_id == 0
    assert vocabulary.unk_id == 1
    assert vocabulary.characters == tuple(sorted(set("https://z.example/Ba")))
    assert list(vocabulary.character_to_id) == list(vocabulary.characters)
    assert tuple(vocabulary.character_to_id.values()) == tuple(
        range(2, len(vocabulary.characters) + 2)
    )
    with pytest.raises(TypeError):
        vocabulary.character_to_id["~"] = 99


def test_vocabulary_is_independent_of_training_iteration_order():
    urls = ["https://z.example/B", "https://a.example/a"]

    assert character_sequence.build_character_vocabulary(
        iter(urls)
    ) == character_sequence.build_character_vocabulary(reversed(urls))


def test_encoding_maps_validation_only_characters_to_unk_and_right_pads():
    vocabulary = character_sequence.build_character_vocabulary(
        ["https://train.example/a"]
    )

    encoded = character_sequence.encode_character_url(
        "https://train.example/~", vocabulary
    )
    normalized = character_sequence.normalize_character_url("https://train.example/~")

    assert encoded.length == len(normalized)
    assert len(encoded.token_ids) == character_sequence.MAX_SEQUENCE_LENGTH
    assert len(encoded.padding_mask) == character_sequence.MAX_SEQUENCE_LENGTH
    assert encoded.token_ids[normalized.index("~")] == vocabulary.unk_id
    assert encoded.token_ids[: encoded.length].count(vocabulary.pad_id) == 0
    assert encoded.token_ids[encoded.length :] == (vocabulary.pad_id,) * (
        character_sequence.MAX_SEQUENCE_LENGTH - encoded.length
    )
    assert encoded.padding_mask == (False,) * encoded.length + (True,) * (
        character_sequence.MAX_SEQUENCE_LENGTH - encoded.length
    )


def test_vocabulary_does_not_learn_characters_from_dropped_middle():
    raw_url = "https://example.test/" + "a" * 180 + "~" + "b" * 80

    vocabulary = character_sequence.build_character_vocabulary([raw_url])

    assert "~" not in vocabulary.character_to_id


def test_vocabulary_json_round_trip_is_strict_and_deterministic():
    vocabulary = character_sequence.build_character_vocabulary(
        ["https://example.test/β"]
    )

    serialized = vocabulary.to_json()
    restored = character_sequence.CharacterVocabulary.from_json(serialized)

    assert restored == vocabulary
    assert restored.to_json() == serialized
    expected = {
        "character_ids_start": 2,
        "characters": list(vocabulary.characters),
        "max_sequence_length": 256,
        "normalization": "canonical-url-v1+utf8-percent-ascii-v1",
        "pad_id": 0,
        "schema_version": 1,
        "unk_id": 1,
    }
    assert json.loads(serialized) == expected
    assert serialized == (
        json.dumps(expected, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )


@pytest.mark.parametrize(
    "raw_url",
    (
        None,
        "",
        "example.test/no-scheme",
        "https://example.test/bad escape %xz",
        "https://example.test/line\nbreak",
        "https://example.test/\ud800",
    ),
)
def test_normalization_rejects_invalid_raw_urls(raw_url):
    with pytest.raises(character_sequence.CharacterSequenceError):
        character_sequence.normalize_character_url(raw_url)


@pytest.mark.parametrize("training_urls", ([], "https://example.test/"))
def test_vocabulary_rejects_empty_or_non_collection_training_input(training_urls):
    with pytest.raises(character_sequence.CharacterSequenceError):
        character_sequence.build_character_vocabulary(training_urls)


@pytest.mark.parametrize(
    "payload",
    (
        "{}",
        '{"schema_version":1,"normalization":"wrong"}',
        (
            '{"character_ids_start":2,"characters":["b","a"],'
            '"max_sequence_length":256,'
            '"normalization":"canonical-url-v1+utf8-percent-ascii-v1",'
            '"pad_id":0,"schema_version":1,"unk_id":1}'
        ),
        (
            '{"character_ids_start":2,"characters":["é"],'
            '"max_sequence_length":256,'
            '"normalization":"canonical-url-v1+utf8-percent-ascii-v1",'
            '"pad_id":0,"schema_version":1,"unk_id":1}'
        ),
        (
            '{"character_ids_start":2,"characters":["a"],'
            '"max_sequence_length":256,'
            '"normalization":"canonical-url-v1+utf8-percent-ascii-v1",'
            '"pad_id":false,"schema_version":true,"unk_id":1}'
        ),
        (
            '{"character_ids_start":2,"characters":["a"],'
            '"characters":["b"],"max_sequence_length":256,'
            '"normalization":"canonical-url-v1+utf8-percent-ascii-v1",'
            '"pad_id":0,"schema_version":1,"unk_id":1}'
        ),
    ),
)
def test_vocabulary_json_rejects_schema_drift(payload):
    with pytest.raises(character_sequence.CharacterSequenceError):
        character_sequence.CharacterVocabulary.from_json(payload)


def test_vocabulary_and_encoding_inputs_are_exact_types():
    with pytest.raises(character_sequence.CharacterSequenceError):
        character_sequence.CharacterVocabulary(())
    with pytest.raises(character_sequence.CharacterSequenceError):
        character_sequence.CharacterVocabulary(("a", "a"))
    with pytest.raises(character_sequence.CharacterSequenceError):
        character_sequence.CharacterVocabulary(("ab",))

    character_sequence.CharacterVocabulary(("a",))
    with pytest.raises(character_sequence.CharacterSequenceError):
        character_sequence.encode_character_url("https://example.test/", object())

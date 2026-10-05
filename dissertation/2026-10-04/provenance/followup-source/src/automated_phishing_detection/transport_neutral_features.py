"""Scheme-invariant follow-up representation; never an original-model input."""

from .phiusiil import PreparationError, canonicalize_url
from .url_features import FEATURE_NAMES as RAW_URL_FEATURE_NAMES
from .url_features import FeatureExtractionError, extract_url_features

REPRESENTATION_ID = "transport-neutral-structural-v1"
FEATURE_NAMES = tuple(
    f"transport_neutral_{name}" for name in RAW_URL_FEATURE_NAMES if name != "is_https"
)


def transport_neutral_model_input(raw_url: object) -> str:
    """Use a fixed modeling prefix, without asserting or changing transport."""
    try:
        canonicalize_url(raw_url)
    except PreparationError as error:
        raise FeatureExtractionError(str(error)) from error
    return "http" + raw_url[raw_url.index(":") :]


def extract_transport_neutral_features(raw_url: object) -> tuple[float, ...]:
    """Extract 24 features after uniform scheme removal from modeled information."""
    values = extract_url_features(transport_neutral_model_input(raw_url))
    return tuple(
        value
        for name, value in zip(RAW_URL_FEATURE_NAMES, values, strict=True)
        if name != "is_https"
    )

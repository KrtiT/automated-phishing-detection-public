# Dataset retrieval correction 1

October 1, 2026, before any new benchmark rows were received or scored.
The first eligibility attempt received HTTP 403 from the publisher download
route. Its intent, failure and console output are preserved under
`population-admission-v1/` and `population-admission-console.txt`.

A normal HEAD request to that same public route returned a publisher-issued
redirect to its public S3 object. A HEAD request to that object returned 200 and
the expected 3,661,166-byte CSV metadata. No authentication or private endpoint
is involved. The next attempt uses a descriptive User-Agent rather than Python's
default, and a new output directory; all publisher identity/hash checks and the
scientific specification remain unchanged. No alternate dataset or observation-
dependent change is introduced. A failed data transfer is not an adverse model
result or a measured HTTP-study retry.

The inspected arXiv article references repository Version 2 whereas the pinned
release is Version 3. Its source-collection account supplies background, not
proof of identical versioned files or verified per-row provenance. The exact
Version 3 metadata/hash remains the source identity for this comparison.

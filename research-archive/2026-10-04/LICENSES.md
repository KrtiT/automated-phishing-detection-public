# Dataset attribution and source identities

Repository software uses the root MIT license. That license does **not** replace
the original licenses of bundled third-party datasets or the Public Suffix List.
No upstream author or publisher endorsement is implied.

| Resource | Attribution and source | License |
|---|---|---|
| PhiUSIIL Phishing URL Dataset | Arvind Prasad and Shalini Chandra; [UCI dataset 967](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| PhishVN v3.1.0 open bundle | Thai Nguyen Vu, University of Transport and Communications, Campus in Ho Chi Minh City (UTC2); [Mendeley version 4](https://doi.org/10.17632/b97hxbxtpd.4) | CC BY 4.0 data; MIT accompanying source, as stated in the bundled LICENSE and CITATION |
| Phishing websites features, benchmark B | Abdelhakim Hannousse and Salima Yahiouche; [Mendeley version 3](https://doi.org/10.17632/c2gw7fy2j4.3) | CC BY 4.0 |
| Public Suffix List | Public Suffix List contributors; [publicsuffix/list](https://github.com/publicsuffix/list), commit `0f1fa47ec45056a19c2fdcd32a08442de9715d12` | [MPL 2.0](https://mozilla.org/MPL/2.0/); license header retained |

PhishVN also credits the NCSC Tin Nhiem Mang source, its trusted registry and
Tranco; the original open-bundle attribution and citation files are preserved.
No private PII mapping or gated phishing HTML is redistributed.

## Frozen byte identities

| File | SHA-256 |
|---|---|
| PhiUSIIL publisher ZIP | `0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a` |
| PhiUSIIL CSV | `a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6` |
| Corrected PhiUSIIL train partition | `575f2fb13a0766020e29d78bf8e633a185b381abde7060bdd1ed04cc4a5e38a0` |
| Corrected PhiUSIIL validation partition | `970c6568a6400a1fc265b7809ef7bd9d1c297632799cbb88801313bc34ac415a` |
| PhishVN open ZIP | `308351e9d0c0fca13a81f2f63524a7dee08ccba6c775848d125b3da5525d0ab5` |
| `dataset_B_05_2020.csv` | `21093e2902e5441c86a6daf95e86e7c332046e477fdf109a579d7bd81e586d6c` |
| Public Suffix List | `65365c4c9a4a6f746d53aadc758ab6b08aa10bb1379fea8ac353e381bca4b62e` |
| Accepted RF correction-v2 `model.json` | `fb134cfb5da65fb5d16b1503595c69e616ebef6699de6f27ef993a0e8ee3a17f` |

These are file hashes, not archive hashes. The archive inventory binds each
identity to its full path. Dataset derivations—label mapping, domain-disjoint
partitioning, exclusions, the disclosed `url_norm` input amendment and the D
population admission—are research changes, not publisher originals. The raw
publisher files remain unchanged. Method contracts and partition manifests
describe those changes; inventories identify every retained derivative separately.

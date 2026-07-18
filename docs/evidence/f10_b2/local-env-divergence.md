# Local workstation (cp314) vs pinned production closure (cp311) — 2026-07-18

Interpreter that produced the 1,685-test green evidence: `C:\Python314\python.exe` (Python 3.14.0).
Production container: `python:3.11.15-slim-bookworm` resolving `requirements.txt`.

| Package | Pinned closure (production) | Local test env |
|---|---|---|
| aiohappyeyeballs | 2.7.1 | 2.6.1 |
| aiohttp | 3.14.1 | 3.13.3 |
| anyio | 4.14.2 | 4.11.0 |
| attrs | 26.1.0 | 25.4.0 |
| certifi | 2026.6.17 | 2025.11.12 |
| cffi | 2.1.0 | 2.0.0 |
| charset-normalizer | 3.4.9 | 3.4.4 |
| click | 8.4.2 | 8.3.1 |
| cryptography | 49.0.0 | 46.0.3 |
| fastapi | 0.109.2 | 0.135.1 |
| filelock | 3.31.0 | 3.29.0 |
| fsspec | 2026.6.0 | 2026.4.0 |
| google-ai-generativelanguage | 0.4.0 | (absent locally) |
| google-api-core | 2.25.2 | (absent locally) |
| google-auth | 2.56.0 | 2.48.0 |
| google-generativeai | 0.4.1 | (absent locally) |
| googleapis-common-protos | 1.75.0 | (absent locally) |
| greenlet | 3.2.5 | 3.3.2 |
| grpcio | 1.82.1 | (absent locally) |
| grpcio-status | 1.62.3 | (absent locally) |
| hf-xet | 1.5.2 | (absent locally) |
| huggingface-hub | 1.24.0 | (absent locally) |
| idna | 3.18 | 3.11 |
| importlib-metadata | 9.0.0 | (absent locally) |
| jiter | 0.16.0 | 0.13.0 |
| jsonpointer | 3.1.1 | 3.0.0 |
| jsonschema | 4.26.0 | (absent locally) |
| jsonschema-specifications | 2025.9.1 | (absent locally) |
| langchain | 0.1.20 | 1.2.10 |
| langchain-community | 0.0.38 | 0.4.1 |
| langchain-core | 0.1.52 | 1.2.17 |
| langchain-google-genai | 0.0.11 | 4.2.1 |
| langchain-openai | 0.1.1 | 1.1.10 |
| langchain-text-splitters | 0.0.2 | 1.1.1 |
| langsmith | 0.1.147 | 0.7.12 |
| litellm | 1.44.10 | (absent locally) |
| multidict | 6.7.1 | 6.7.0 |
| numpy | 1.26.4 | 2.3.5 |
| openai | 1.109.1 | 2.24.0 |
| orjson | 3.11.9 | 3.11.7 |
| packaging | 23.2 | 25.0 |
| pandas | 2.1.4 | 2.3.3 |
| patsy | 1.0.2 | (absent locally) |
| propcache | 0.5.2 | 0.4.1 |
| proto-plus | 1.28.1 | (absent locally) |
| protobuf | 4.25.9 | 6.33.1 |
| psycopg | 3.2.1 | 3.3.3 |
| psycopg-binary | 3.2.1 | 3.3.3 |
| pyasn1 | 0.6.4 | 0.6.2 |
| pycparser | 3.0 | 2.23 |
| pydantic | 2.9.2 | 2.12.4 |
| pydantic-core | 2.23.4 | 2.41.5 |
| pyjwt | 2.9.0 | 2.10.1 |
| python-dotenv | 1.0.1 | 1.2.1 |
| pytz | 2026.2 | 2025.2 |
| referencing | 0.37.0 | (absent locally) |
| regex | 2026.7.10 | 2026.2.28 |
| requests | 2.34.2 | 2.32.5 |
| rpds-py | 2026.6.3 | (absent locally) |
| scipy | 1.11.4 | 1.16.3 |
| sqlalchemy | 2.0.44 | 2.0.48 |
| sqlglot | 25.25.0 | 29.0.1 |
| starlette | 0.36.3 | 0.52.1 |
| statsmodels | 0.14.1 | (absent locally) |
| tenacity | 8.2.3 | 9.1.4 |
| tiktoken | 0.11.0 | 0.12.0 |
| tokenizers | 0.23.1 | (absent locally) |
| tqdm | 4.69.0 | 4.67.3 |
| typing-extensions | 4.16.0 | 4.15.0 |
| tzdata | 2026.3 | 2025.2 |
| urllib3 | 2.7.0 | 2.5.0 |
| uvicorn | 0.27.1 | 0.41.0 |
| websockets | 16.1.1 | 15.0.1 |
| wrapt | 2.2.2 | 2.1.1 |
| yarl | 1.24.2 | 1.22.0 |
| zipp | 4.1.0 | (absent locally) |

## Local-env packages still carrying OSV advisories (GHSA ids)

- `aiohttp==3.13.3`: GHSA-2fqr-mr3j-6wp8, GHSA-2vrm-gr82-f7m5, GHSA-3wq7-rqq7-wx6j, GHSA-4fvr-rgm6-gqmc, GHSA-4m7w-qmgq-4wj5, GHSA-63hf-3vf5-4wqf, GHSA-63hw-fmq6-xxg2, GHSA-966j-vmvw-g2g9, GHSA-9x8q-7h8h-wcw9, GHSA-c427-h43c-vf67, GHSA-g3cq-j2xw-wf74, GHSA-hcc4-c3v8-rx92, GHSA-hg6j-4rv6-33pg, GHSA-hpj7-wq8m-9hgp, GHSA-jg22-mg44-37j8, GHSA-m5qp-6w8w-w647, GHSA-m6qw-4cw2-hm4m, GHSA-mwh4-6h8g-pg8w, GHSA-p998-jp59-783m, GHSA-w2fm-2cpv-w7v5, GHSA-xcgm-r5h9-7989
- `cryptography==46.0.3`: GHSA-537c-gmf6-5ccf, GHSA-m959-cc7f-wv43, GHSA-p423-j2cm-9vmq, GHSA-r6ph-v2qm-q3c2
- `curl-cffi==0.13.0`: GHSA-qw2m-4pqf-rmpp
- `idna==3.11`: GHSA-65pc-fj4g-8rjx
- `langchain-classic==1.0.1`: GHSA-3644-q5cj-c5c7
- `langchain-core==1.2.17`: GHSA-926x-3r5x-gfhw, GHSA-pjwx-r37v-7724, GHSA-qh6h-p6c9-ff54
- `langchain-openai==1.1.10`: GHSA-r7w7-9xr2-qq2r
- `langchain-text-splitters==1.1.1`: GHSA-fv5p-p927-qmxr
- `langchain==1.2.10`: GHSA-gr75-jv2w-4656
- `langgraph-checkpoint==4.0.1`: GHSA-fjqc-hq36-qh5p
- `langgraph-sdk==0.3.9`: GHSA-w39p-vh2g-g8g5
- `langsmith==0.7.12`: GHSA-3644-q5cj-c5c7, GHSA-f4xh-w4cj-qxq8, GHSA-rr7j-v2q5-chgv
- `lxml==6.0.2`: GHSA-vfmq-68hx-4jfw
- `protobuf==6.33.1`: GHSA-7gcm-g887-7qv7
- `pyasn1==0.6.2`: GHSA-jr27-m4p2-rc6r
- `pydantic-settings==2.13.1`: GHSA-4xgf-cpjx-pc3j
- `pygments==2.19.2`: GHSA-5239-wwwm-4pmq
- `pyjwt==2.10.1`: GHSA-752w-5fwx-jx9f, GHSA-993g-76c3-p5m4, GHSA-fhv5-28vv-h8m8, GHSA-jq35-7prp-9v3f, GHSA-w7vc-732c-9m39, GHSA-xgmm-8j9v-c9wx
- `pytest==9.0.2`: GHSA-6w46-j5rx-g56g
- `python-dotenv==1.2.1`: GHSA-mf9w-mj56-hr94
- `requests==2.32.5`: GHSA-gc5v-m9x4-r6x2
- `soupsieve==2.8`: GHSA-2wc2-fm75-p42x, GHSA-836r-79rf-4m37
- `starlette==0.52.1`: GHSA-82w8-qh3p-5jfq, GHSA-86qp-5c8j-p5mr, GHSA-jp82-jpqv-5vv3, GHSA-wqp7-x3pw-xc5r, GHSA-x746-7m8f-x49c
- `torch==2.11.0`: GHSA-rrmf-rvhw-rf47
- `urllib3==2.5.0`: GHSA-2xpw-w6gg-jr37, GHSA-38jv-5279-wg99, GHSA-gm62-xv2j-4w53, GHSA-qccp-gfcp-xxvc

# CSI500 / CSI1000 data audit and go/no-go rules

Status (2026-07-13): the local formal-data build and Gates A--F pass. The
checked-in `market_snapshots.example.csv` remains explicitly synthetic; formal
derived artifacts are reproducibly generated under ignored `data/derived/` and
raw market data are not redistributed.

The frozen formal panel has 72 rows: 36 monthly observations for CSI500 and 36
for CSI1000 from 2023-01 through 2025-12, arranged as 12 half-year episodes of
six rounds. Public CSI500 and CSI1000 spot, returns, and realized volatility use
the index series `000905.SH` and `000852.SH`, respectively. `510500.SH` is never
used as public CSI500 spot; it is used only as the explicitly labelled
`sse_510500_etf_iv_proxy` inside the CSI500 volatility pipeline.

## Required minimum

Formal experiments require, for each 2023--2025 month-end:

- CSI500 or CSI1000 spot/close;
- at least one volatility input available at that date;
- non-empty source provenance;
- only information observable on or before that date.

Option IV is settlement-only with positive open interest. CSI500 uses the SSE
510500 ETF option chain collected through the authorized Tushare bridge, exact
`OP510500.SH` master whitelist, and normal (`M`)
contracts only; adjusted (`A`) contracts are excluded. CSI1000 uses CFFEX MO
contracts from official CFFEX historical ZIP archives, joined exactly from
daily contract symbol to the `OP000852.SH` master.
Candidate dates are month-end and the preceding two market trading days, with
spot, SHIBOR, and option observations required to share the selected date.

The final full build has IV on 69/72 rows (95.83%) and all 1m/3m/6m tenors on
56/72 rows (77.78%). Coverage is 33/36 for CSI500 and 36/36 for CSI1000; tenor
coverage is 69/72 for 1m, 62/72 for 3m, and 56/72 for 6m. The remaining three
rows are explicitly labelled `rv_fallback`. All 69 selected chains use the
same-day (`t0`) observation.

Risk-free rates use the lag-controlled SHIBOR curve and publish its 91-day
continuous rate. The preferred ChinaBond government yield curve endpoint
`yc_cb` could not be used because the configured account lacks access
permission. Index carry
is independent of the option pipeline: CSI500 uses IC and CSI1000 uses IM
same-day futures settlements and active futures masters to infer 91-day carry.
Carry is observed on all 72 rows, with no zero fallback.

## Official source hierarchy

1. CSI1000: CFFEX MO product and contract-level historical service:
   <https://www.cffex.com.cn/zz1000gzqq/> and
   <https://www.cffex.com.cn/lssjfw/>.
2. CSI500: SSE 510500 ETF option contract documentation and exchange historical
   files: <https://www.sse.com.cn/assortment/options/contract/c/c_20230303_5717361.shtml>.
3. If official public files do not expose a stable expired contract chain, use a
   licensed Wind/Choice/CFFEX Level-1 export without redistributing raw records.
4. If no licensed chain is available by the data freeze, use public index/ETF
   closes and realized volatility. Do not synthesize or backfill a historical IV
   surface and describe it as observed data.

ETF fee changes and dividend-related contract adjustments must be date-aware.
An example 510500 fee adjustment is documented at
<https://www.sse.com.cn/disclosure/fund/announcement/c/new/2024-11-20/510500_20241120_WGM2.pdf>.
An example exchange option adjustment after ETF distribution is available at
<https://www.szse.cn/option/notice/notify/t20251121_617382.html>.

## Acceptance gates (current result)

- Gate A -- **PASS**: 72 rows, 12 x 6 episodes, complete per-row provenance,
  valid raw and derived manifests.
- Gate B -- **PASS**: every `max_input_date <= as_of`, strict 20/60-return and
  126-level windows are recorded, and two consecutive full rebuilds produce
  identical snapshot and provenance hashes.
- Gate C -- **PASS**: 69 rows use observed proxy/index IV and three use the
  explicit realized-volatility fallback; source labels are machine-audited.
- Gate D -- **PASS**: the exact SSE whitelist contains 1,614 contracts, with
  zero invalid masters and zero active-join failures; selected contracts are
  version `M` only (346 `A` masters excluded).
- Gate E -- **PASS**: all 322 selected expiry points have DTE 7--365,
  `F/S` in `[0.75, 1.25]`, parity MAD at most 1%, parity range at most 3%, IV in
  `[0.01, 1.5]`, settlement pricing, and explicit contract provenance.
- Gate F -- **PASS for local research**: authorization is recorded in
  `data/reference/license_decision.json`. Raw redistribution is false; raw
  directories and `.env` are gitignored, so release scope is derived features,
  code, hashes, and acquisition instructions only.

Failure of an individual option-chain row triggers the pre-registered
realized-volatility fallback. It does not change the public index spot, futures
carry, episode layout, or held-out periods. Reproduce the complete audit with
`python -m stochopia.formal_gate_audit`.

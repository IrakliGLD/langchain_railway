# Exchange Transition

## Source Scope

Primary source: final Electricity Market Concept Article 17^4.

Related source:
- Transitory Market Rules Article 10^2: exchange trading authorization refers to the Article 17^4 buyer/seller lists
- Transitory Market Rules Articles 13-14: monthly balancing settlement continues during the exchange transition

---

## 1. Transition Period

Article 17^4 governs wholesale electricity trade from **July 1, 2024 to July 1, 2027**.

During this period, wholesale electricity can be bought and sold:
- on the exchange under the day-ahead/intraday market rules
- by bilateral direct contract under the 2006 market rules
- through balancing electricity settlement under the 2006 market rules

This is a hybrid period, not the final target market model.

---

## 2. Voluntary Exchange Trading

Exchange trading is voluntary during the Article 17^4 transition.

Important implications:
- the existence of GENEX does not mean all wholesale electricity clears through the exchange
- exchange prices are partial market signals
- direct contracts and ESCO balancing settlement remain active
- thin exchange volumes should not be treated as the whole-market price

Use this file for "who can trade on the exchange", "is exchange trading mandatory", and "how exchange affects balancing" questions.

---

## 3. Monthly Balancing Still Applies

Article 17^4 keeps the monthly balancing logic during the transition.

Key rule:
- balancing electricity price and quantity are calculated monthly
- if a participant trades hourly on the exchange or by bilateral direct contract, the hourly traded electricity is counted in that participant's total monthly traded quantity

Practical implication:

```text
monthly residual balancing quantity
  = actual monthly physical quantity
    minus exchange-traded quantity
    minus registered direct-contract quantity
```

Do not treat exchange trading after July 2024 as a replacement for monthly balancing settlement.

---

## 4. Eligible Buyers

Only registered wholesale-market participants named in Article 17^4(6) may buy on the exchange during the transition.

The buyer list includes, in substance:
- direct customers buying for their own enterprise consumption
- universal service supplier
- public service supplier
- free supplier
- last-resort supplier
- transmission system operator / Dispatch Licensee for losses and transit-related needs
- distribution system operator for distribution losses
- other buyer categories expressly named in Article 17^4(6)

Do not answer that all end-users can buy on GENEX. Eligibility is limited to the Article 17^4 buyer categories and registration status.

---

## 5. Eligible Sellers

Only wholesale-market participants named in Article 17^4(7) may sell on the exchange during the transition.

The seller side mainly covers eligible generation/project categories that are permitted to sell on the exchange during the transition. The list is not equivalent to "all generators." Support-scheme and PSO-bound plants may be limited by their support period, mandatory ESCO-sale obligation, or other regulatory status.

For a precise seller answer, cite Article 17^4(7) and avoid broad generalizations.

---

## 6. Link to Direct Contracts

Direct contracts continue in parallel with exchange trading.

Article 10^2 of the Transitory Market Rules points back to Article 17^4:
- exchange sellers are the Article 17^4(7) participants
- exchange buyers are the Article 17^4(6) participants

Use `direct_contracts.md` for contract registration and `exchange_transition.md` for exchange eligibility and transition-period market design.

---

## 7. Link to Target Model

The Article 17^4 transition ends on **July 1, 2027**, the planned target-model date.

Target model expectations:
- self-dispatch
- organized day-ahead, intraday, balancing, and ancillary service markets
- hourly trading and hourly imbalance responsibility
- BRP/BSP roles
- transparent competitive price formation

Forecasts or policy answers that cross July 1, 2027 must flag the structural break between:
- current/transitional monthly weighted-average balancing model
- target hourly self-dispatch market model

---

## 8. Data Interpretation

In internal data:
- exchange segment appears only from July 2024
- exchange volumes are not the same as total wholesale trade
- exchange prices should not be treated as a complete system marginal price during 2024-2027
- balancing-price analysis remains governed by `balancing_price.md`

For questions about "GENEX impact on price", answer:
- exchange adds a competitive hourly trading layer
- it reduces residual balancing exposure for participants that trade there
- it does not yet replace monthly ESCO balancing-price formation

---

## 9. Answering Guidance

Use `exchange_transition.md` when the user asks about:
- GENEX after July 2024
- exchange eligibility
- voluntary exchange trading
- day-ahead/intraday transition
- Article 17^4
- July 2024 to July 2027 market transition
- whether exchange replaces balancing

When answering:
1. State the transition period exactly: July 1, 2024 to July 1, 2027.
2. Say exchange trading is voluntary.
3. Explain that monthly balancing continues.
4. Mention that only Article 17^4 eligible buyers/sellers may trade.
5. If the question asks about final target model, distinguish it from the transition.

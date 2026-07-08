# Direct Contracts

## Source Scope

Primary source: final Transitory Market Rules, Chapter III:
- Article 8: parties to direct contracts
- Article 9: contract registration with the Dispatch Licensee and Market Operator
- Article 9^1: import/export contract registration with the Dispatch Licensee
- Article 10: contract and registration termination
- Article 10^1: project-company direct contract registration
- Article 10^2: exchange trading authorization during the Concept transition

Related sources:
- Electricity Market Concept Article 17^4: transitional wholesale trading from July 1, 2024 to July 1, 2027
- Transitory Market Rules Article 13: which contracted or residual quantities count as balancing electricity
- Transitory Market Rules Article 14: how direct-contract prices enter ESCO buy-side and sell-side balancing-price formulas

---

## 1. Definition

A direct contract is a bilateral electricity purchase/sale contract used in the wholesale market. It is separate from:
- exchange trading on the day-ahead or intraday market
- residual balancing electricity settled through ESCO
- regulated tariff setting by GNERC

Direct contracts can be commercial bilateral contracts, mandatory direct contracts required by law/government acts, support-scheme purchase contracts, import/export contracts, or project-company registrations depending on the article and context.

---

## 2. Who Can Use Direct Contracts

Article 8 allows direct contracts between eligible wholesale-market participants, especially qualified enterprises and licensees/participants that have the legal right to participate in wholesale electricity trade.

Common practical parties include:
- producers
- importers
- exporters
- wholesale suppliers or traders
- direct customers
- transmission, dispatch, or distribution licensees where the rules permit it
- ESCO / Market Operator roles where mandatory purchase or public-service logic applies

Do not answer that any retail customer can freely sign a wholesale direct contract. The party must be a qualified/eligible wholesale participant under the relevant rule.

---

## 3. Contract Content

A direct contract or contractual application should identify the commercial and settlement terms needed for registration and dispatch accounting, including:
- contracted electricity quantity or the method for determining quantity
- monthly indicative quantity
- unit price
- start and end time/date of supply
- delivery conditions
- producer/plant name when the seller is a generator
- termination conditions
- contract number and signing date

Quantity can be expressed as a fixed value or by a formula. The contract/application terms matter because monthly residual balancing quantities are calculated after contracted and exchange-traded quantities are netted out.

---

## 4. Registration Rule

Direct contracts are not just private side agreements for settlement purposes. They must be registered with the Dispatch Licensee and the Market Operator under Article 9.

Key rule:
- registration is required before the contract can be recognized in the market settlement process
- registration is made through the contract registration electronic portal where applicable
- the contractual application is part of the contract registration package
- if the registered application and the contract copy conflict, the application terms are operationally important for market settlement

For import/export contracts, Article 9^1 adds a separate registration requirement with the Dispatch Licensee. See `cross_border_trade.md`.

---

## 5. Effect on Balancing Electricity

Direct contracts are central to the current monthly balancing model:

```text
balancing exposure = actual generation/consumption/import/export
                     minus registered direct-contract quantities
                     minus relevant exchange-traded quantities
```

Practical implications:
- contracted generation usually reduces the seller's residual balancing sales to ESCO
- contracted consumption usually reduces the buyer's residual balancing purchase from ESCO
- import/export direct-contracted electricity is handled separately and is not automatically part of the sell-side balancing price
- uncontracted residual quantities fall back into the balancing mechanism under Article 13

For price questions, do not treat the direct-contract price as `p_bal_gel`. `p_bal_gel` is the monthly sell-side weighted-average balancing price, not a bilateral contract price.

---

## 6. Mandatory Direct Contracts and Support Schemes

Some direct contracts are mandatory or government-backed rather than freely negotiated commercial contracts.

Important cases:
- PPA/CfD support-scheme electricity sold to ESCO is linked to mandatory direct-contract logic
- Article 14(1)(1) states that ESCO purchases electricity covered by Article 13(5) and 13(5^1) direct-contract cases at the price defined in the direct contract
- support-scheme contract prices are usually confidential and should not be presented as public tariffs

For support-scheme analysis, combine this file with `cfd_ppa.md` and `balancing_price.md`.

---

## 7. Termination and Changes

Article 10 governs termination of direct-contract registration and related contract changes.

Practical answer rule:
- a contract ending commercially is not enough for market treatment unless the required termination/change notification is handled through the market registration process
- cancellation or change affects future monthly settlement quantities
- do not retroactively remove already-settled balancing quantities unless the source explicitly says the registration correction is retroactive

---

## 8. Project Companies

Article 10^1 creates a special registration pathway for project companies, especially where a plant or project has a government-backed development/support arrangement.

Use this topic for questions about:
- preregistration or preliminary registration of direct contracts for new projects
- guaranteed purchase or project-company contracts
- when a project company can keep or lose preregistered contract status

Do not confuse a project-company preregistration with ordinary exchange participation.

---

## 9. Exchange Link

Article 10^2 points to the Concept Article 17^4 participant lists for exchange trading authorization. During the July 1, 2024 to July 1, 2027 transition:
- only the eligible buyers and sellers named in Concept Article 17^4 may trade on the exchange
- exchange trading is voluntary
- direct contracts and balancing settlement continue in parallel

For exchange-eligibility questions, use `exchange_transition.md`.

---

## 10. Answering Guidance

Use `direct_contracts.md` when the user asks about:
- direct contract
- bilateral contract
- contractual application
- contract registration
- import/export contract registration
- project company contract
- guaranteed purchase contract
- how direct contracts affect balancing settlement

When answering:
1. First identify whether the question is about ordinary bilateral trade, import/export, support-scheme purchase, or project-company registration.
2. State that registration is required for market settlement recognition.
3. Explain the link to balancing: registered direct-contract quantities reduce residual balancing quantities.
4. For prices, distinguish direct-contract price from ESCO's monthly sell-side balancing price.

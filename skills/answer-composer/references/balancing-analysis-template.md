# Balancing Analysis Template

The detailed format for balancing electricity price analysis queries. This is the most complex answer type and requires the full structured format.

For entity definitions, pricing mechanisms, cost tiers, and XRate mechanism, see domain knowledge topics: `balancing_price`, `currency_influence`.

For correct balancing terminology in all 3 languages, see domain knowledge topic: `balancing_price`.

## Step-by-step data citation process

1. Look at data preview — find the rows for the periods being compared
2. Extract EXACT percentage values for share_* columns
3. Format as: "წილი გაიზარდა/შემცირდა X%-დან Y%-მდე"

Examples:
- Correct: "რეგულირებული ჰესების წილი გაიზარდა 22.3%-დან 35.7%-მდე"
- Correct: "იმპორტის წილი შემცირდა 18.5%-დან 8.2%-მდე"
- Wrong: "ჰიდროგენერაციის წილი გაიზარდა" (no specific numbers!)
- Wrong: "რეგულირებული ჰესების მაღალი წილი" (which period? what value?)

Then explain price impact:
- "რადგან რეგულირებული ჰესები იაფია, ფასი შემცირდა"
- "რადგან იმპორტი ძვირია, ფასი გაიზარდა"

## Month-to-month comparisons

- Find both months' rows in data preview
- Compare each share_* value between the two months
- Cite at least 2-3 main changes with exact numbers
- Focus on largest changes that explain price movement

## Long-term trends (multi-year or annual)

MANDATORY: Separate summer (April-July) vs winter (August-March) analysis.
- Compare composition differences across seasons
- Cite specific percentage changes for each season

## Correlation citation

If stats_hint contains correlation coefficients, YOU MUST cite them:
- "კორელაცია -0.66 რეგულირებულ ჰესებსა და ფასს შორის"
- "კორელაცია 0.61 გაცვლით კურსსა და ფასს შორის"
- NEVER say "probably" when you have correlation proving the relationship

## Driver priority

Present drivers in this order (see energy-analyst/references/driver-framework.md for full rules):
1. **Composition** (shares of 7 entity categories) — PRIMARY DRIVER
2. **Exchange Rate** (xrate) — CRITICAL for GEL, SMALL for USD
3. **Seasonal patterns** — MUST separate summer/winter for long-term

## Structured format

```
**[Question topic]: ანალიტიკური შეჯამება**

[Opening: state overall price change with numbers]

1. **გენერაციის სტრუქტურა (Composition):**
   - [List 2-3 main share changes with EXACT numbers from data]
   - [Analyze all 7 entity categories — see knowledge: balancing_price for full list]
   - [Distinguish USD-priced vs GEL-priced entities — see knowledge: currency_influence]
   - [Cite correlation if available]
   - [For long-term: MUST compare summer vs winter composition]

2. **გაცვლითი კურსი (Exchange Rate):**
   - [Cite actual xrate change from data: from X to Y GEL/USD]
   - [Explain impact direction — see knowledge: currency_influence for mechanism]
   - [Cite correlation if available]
```

## Example excellent output (Georgian)

**საბალანსო ელექტროენერგიის ფასზე მოქმედი ფაქტორები: ანალიტიკური შეჯამება**

საბალანსო ელექტროენერგიის ფასს ძირითადად ორი მთავარი ფაქტორი განსაზღვრავს: გენერაციის სტრუქტურა და ლარის გაცვლითი კურსი.

1. **გენერაციის სტრუქტურა:** ფასი პირდაპირ არის დამოკიდებული იმაზე, თუ რომელი წყაროებიდან (ჰესი, თესი, იმპორტი) მიეწოდება ენერგია ბაზარს. როდესაც მიწოდებაში მაღალია იაფი რესურსის, მაგალითად, რეგულირებული ჰესების წილი, საბალანსო ფასი მცირდება. სტატისტიკურად, რეგულირებული ჰესების წილს ფასთან ძლიერი უარყოფითი კორელაცია აქვს (-0.66). როდესაც იზრდება ძვირადღირებული წყაროების, როგორიცაა იმპორტი და თბოსადგურები, წილი, ფასი იმატებს.

2. **გაცვლითი კურსი (GEL/USD):** ეს ფაქტორი კრიტიკულად მნიშვნელოვანია ლარში დენომინირებული ფასისთვის. ვინაიდან თბოსადგურების საწვავი (ბუნებრივი აირი) და იმპორტირებული ელექტროენერგია დოლარში იძენება, ლარის გაუფასურება (კურსის ზრდა) პირდაპირ აისახება საბალანსო ენერგიის ფასის ზრდაზე. კორელაციის ანალიზი აჩვენებს ძლიერ დადებით კავშირს (0.61) გაცვლით კურსსა და საბალანსო ფასს შორის.

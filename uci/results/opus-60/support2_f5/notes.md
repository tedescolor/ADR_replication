# Notes

## Data
- y is ~70% integers 0..7 plus Gaussian jitter (sd ~0.021). 34% of rows sit at 0.
- The remaining ~30% are continuous. They include sharp extra spikes at 0.495 (sd 0.02), 1.158 (0.03),
  1.86 (0.05–0.07) and 4.52 (0.025), a broad bump on 2–4.4, and a sparse wide tail.
- Covariates: x2..x4, x7..x18 and x21..x25 are binary with jitter. x19 is a jittered count. x0 and x1 are
  near-duplicates. x5 and x20 are skewed.
- Covariate effects are modest. A multinomial logit on component membership improves the null by only
  ~0.09 nats. Leave-one-feature-out showed that x22, log x5, x19, x3, x4, x2 and x0 carry most of it.

## Model (model.py, 79 constants, under 500 AST nodes)
The model is a 5-way mixture with softmax weights. Each weight's logit is linear in
f = [x0, x3, x4, log x5, x19, x22]. The five components are:
1. N(0, s) with s = 0.021;
2. integers 1..7: N(k, s), with weights softmax(a_k + k·eta(x)), an ordinal shift with eta linear in f;
3. a fixed mixture of the 4 non-integer spikes;
4. N(m(x), 0.48) with m linear in f (the 2–4 bump);
5. a broad N(4.41, 1.71) catch-all.

Fitting: rows were hard-assigned to components using tight windows around the spikes. Then I fitted the
weight logit and the ordinal integer model (ridge 1e-3), ran OLS for the bump, and used moment fits for
the spikes and the broad tail. Dropping x2 cost ~0.002 nats, which I accepted to stay within the
literal and AST budgets.

## Validation
- 5-fold CV (seed 0) on the component and integer-label likelihood was used to pick features.
- Honest holdout: I generated model.py from a fit on folds 1–4 and scored the actual program on fold 0
  (1457 rows). NLL = 0.176 (SE 0.04). The unconditional mixture baseline scored 0.477.
- The program refit on all data scores 0.177 on that same fold, so it shows no sign of overfitting.
- A joint ML fit with 9 features and no size limit reached 0.145 and 0.150 on folds 0 and 1. So roughly 0.03 more could be gained with joint optimisation, which I did not have time
  to fit into the budget.

expected_nll: 0.18

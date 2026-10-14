# Negative results and limits

## Resource guard setup

Initial use of a 6 GiB address-space limit prevented Bend/Lean from reserving
virtual memory. This was a harness setup failure, not numerical evidence or a
candidate performance result. Its logs remain in the baseline evidence. Accepted
runs instead use a sampled aggregate process-group RSS watchdog, identically
configured across compared arms.

## Prefill skipping

Skipping logits that are unused during prompt consumption looked promising for
the resident first-ID clock; bit preservation is a separate correctness gate. The five fresh one-thread, 32-token-prompt finalist pairs initially
cleared the selected native first-ID clock criterion. Five additional independent
pairs produced savings of 7.79%, 11.35%, 11.92%, 4.39% and 7.72%: median 7.79%,
with a 4.39% bootstrap lower bound. That does not clear the required 5% lower
bound. The independent batch is not pooled into the earlier batch to hide the
weaker result. The eight-thread candidate did not clear the gate either.

The native first-ID clock begins after model loading and tokenization. It is
not end-to-end time to decoded text. Independent decoded-text arrival savings
were 1.34–4.66%, median 3.18%. The candidate remains isolated and unselected.
Extended prompt checks do not change this performance decision.

## Loader zero-band reuse

The source change removes repeated zero-band allocation. Fresh A/B medians are
favorable on some loader/process measures, but variation prevents the required
primary-metric acceptance. There is no default change. Further attribution
runs use an instrumented C diagnostic and microprobes; they will not be presented
as production inference latency.

## Shape/layout claims

B's shape sweep uses synthetic kernel fixtures with the real model dimensions.
It does not run the 360M or 1.7B models. C's packed-eight prototype is also kernel
research. Neither may be called a whole-model speedup without isolated integration
and real-model numerical and latency confirmation. Large-head work is deferred
for calibrated resources, not declared impossible.

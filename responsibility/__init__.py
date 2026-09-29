"""Counterfactual responsibility (Hsu et al., IROS 2023) on top of CAT's
DenseTNT traffic prior: how much an agent's actual behaviour reduced its
safety margin to its neighbours compared with what it could otherwise have
done (safety responsibility), and how much its presence changed its
neighbours' intended behaviour (courtesy responsibility).

Ported from the SMART/WOMD implementation in the catk repository
(src/responsibility); see responsibility/README.md.
"""

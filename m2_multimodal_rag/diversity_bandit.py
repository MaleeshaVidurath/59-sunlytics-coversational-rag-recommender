"""
M2 Thompson Sampling Diversity Bandit.

Implements an adaptive Bayesian bandit that dynamically tunes the MMR
diversity parameter λ based on implicit feedback signals observed within
each conversational turn.

Background — MMR tradeoff:
    MMR(i) = λ × relevance(i) − (1−λ) × max_sim(i, already_selected)
    λ = 1.0  →  pure relevance ranking   (show the most relevant items)
    λ = 0.0  →  pure diversity selection  (show the most different items)

A fixed λ (e.g. 0.7) cannot adapt to user behaviour. This bandit models
the optimal λ as a latent Bernoulli probability and maintains a Beta
posterior over it, updated each turn using implicit feedback signals:

    • Rejected items  (exclude_ids)        → user wants MORE diversity  → β increases
    • Retained items  (items_in_context)   → user wants MORE relevance  → α increases

Thompson Sampling:
    Each turn, λ is sampled from the current Beta(α, β) posterior:
        λ ~ Beta(α, β)
    rather than always using the posterior mean. This exploration-exploitation
    balance means the system occasionally tries a more diverse or more relevant
    configuration, naturally discovering what works best for this user.

Prior:
    Beta(7, 3) → E[λ] = 0.70, matching the original fixed λ used before RL.
    This ensures the bandit starts with sensible behaviour on the first turn.

Reference: Thompson (1933) — "On the Likelihood that One Unknown Probability
           Exceeds Another"; applied to recommender exploration in Li et al. (2010).
"""

import numpy as np


class ThompsonSamplingDiversityBandit:

    # Uninformative-but-sensible prior: mean = 7/(7+3) = 0.70
    PRIOR_ALPHA = 7.0
    PRIOR_BETA = 3.0

    # How strongly each feedback signal shifts the posterior
    REJECTION_WEIGHT = 0.8   # Each rejected item → +0.8 to β (more diversity)
    RETENTION_WEIGHT = 0.5   # Each retained item → +0.5 to α (more relevance)

    # Safe operating range — never fully ignore relevance or diversity
    LAMBDA_MIN = 0.50
    LAMBDA_MAX = 0.90

    def __init__(self):
        print("M2 Bandit: Thompson Sampling Diversity Bandit initialised.")
        prior_mean = self.PRIOR_ALPHA / (self.PRIOR_ALPHA + self.PRIOR_BETA)
        print(f"M2 Bandit: Prior Beta({self.PRIOR_ALPHA}, {self.PRIOR_BETA}) "
              f"→ E[λ] = {prior_mean:.2f} (matches original fixed λ)")

    def sample_lambda(self, exclude_count: int, retained_count: int) -> float:
        """
        Samples the MMR λ from a context-updated Beta posterior.

        The posterior is computed fresh each call from the stateless prior
        plus the current turn's implicit feedback signals.  This design
        requires no persistent session state while still producing turn-
        aware adaptive behaviour.

        Args:
            exclude_count  : len(exclude_ids) — items the user rejected this session.
            retained_count : items_in_context count — items the user kept discussing.

        Returns:
            λ ∈ [LAMBDA_MIN, LAMBDA_MAX] sampled via Thompson Sampling.
        """
        # Bayesian update: shift posterior based on observed feedback signals
        alpha = self.PRIOR_ALPHA + retained_count * self.RETENTION_WEIGHT
        beta  = self.PRIOR_BETA  + exclude_count  * self.REJECTION_WEIGHT

        # Thompson Sampling: draw one sample from the posterior
        raw_sample = float(np.random.beta(alpha, beta))

        # Clip to safe operating window
        lambda_val = float(np.clip(raw_sample, self.LAMBDA_MIN, self.LAMBDA_MAX))

        posterior_mean = alpha / (alpha + beta)
        print(f"   [Bandit] Signals — rejected: {exclude_count}, retained: {retained_count}")
        print(f"   [Bandit] Posterior: Beta({alpha:.1f}, {beta:.1f}) "
              f"E[λ]={posterior_mean:.3f} → sampled λ={lambda_val:.3f}")
        return lambda_val

    def expected_lambda(self, exclude_count: int, retained_count: int) -> float:
        """
        Returns the posterior mean E[λ] = α/(α+β) without sampling.
        Useful for logging and unit testing.
        """
        alpha = self.PRIOR_ALPHA + retained_count * self.RETENTION_WEIGHT
        beta  = self.PRIOR_BETA  + exclude_count  * self.REJECTION_WEIGHT
        return float(np.clip(alpha / (alpha + beta), self.LAMBDA_MIN, self.LAMBDA_MAX))


# Singleton — stateless design means one instance serves all requests safely
diversity_bandit = ThompsonSamplingDiversityBandit()

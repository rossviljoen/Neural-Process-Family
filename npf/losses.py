"""Module for all the loss of Neural Process Family."""
import abc
import math

import torch
import torch.nn as nn
from npf.utils.helpers import (
    MultivariateNormalDiag,
    LightTailPareto,
    dist_to_device,
    logcumsumexp,
    sum_from_nth_dim,
)
from torch.distributions.kl import kl_divergence

__all__ = ["CNPFLoss", "ELBOLossLNPF", "PAC2LossLNPF", "PACMLossLNPF", "PAC2TLossLNPF", "SUMOLossLNPF", "NLLLossLNPF", "PACELBOLossLNPF"]


def sum_log_prob(prob, sample):
    """Compute log probability then sum all but the z_samples and batch."""
    # size = [n_z_samples, batch_size, *]
    log_p = prob.log_prob(sample)
    # size = [n_z_samples, batch_size]
    sum_log_p = sum_from_nth_dim(log_p, 2)
    return sum_log_p


class BaseLossNPF(nn.Module, abc.ABC):
    """
    Compute the negative log likelihood loss for members of the conditional neural process (sub-)family.

    Parameters
    ----------
    reduction : {None,"mean","sum"}, optional
        Batch wise reduction.

    is_force_mle_eval : bool, optional
        Whether to force mac likelihood eval even if has access to q_zCct
    """

    def __init__(
            self,
            reduction="mean",
            is_force_mle_eval=True,
            # train_all_data=False,
            eval_use_crossentropy=False,
            beta=1.
    ):
        super().__init__()
        self.reduction = reduction
        self.is_force_mle_eval = is_force_mle_eval
        # self.train_all_data = train_all_data
        self.eval_use_crossentropy = eval_use_crossentropy
        self.beta = beta

    def forward(self, pred_outputs, Y):
        """Compute the Neural Process Loss.

        Parameters
        ----------
        pred_outputs : tuple
            Output of `NeuralProcessFamily`.

        Y_trgt : torch.Tensor, size=[batch_size, *n_trgt, y_dim]
            Set of all target values {y_t}.

        Return
        ------
        loss : torch.Tensor
            size=[batch_size] if `reduction=None` else [1].
        """
        Y_cntxt, Y_trgt = Y['Y_cntxt'], Y['Y_trgt']
        p_yCc, z_samples, q_zCc, q_zCct, p_z, decoder_kl = pred_outputs

        if self.training:
            # if self.train_all_data: # TODO: remove this option
            #     loss = self.get_loss(p_yCc, z_samples, q_zCc, q_zCct, p_z, decoder_kl, torch.cat([Y_cntxt, Y_trgt], dim=1))
            loss = self.get_loss(p_yCc, z_samples, q_zCc, q_zCct, p_z, decoder_kl, Y_trgt)
        else:
            # always uses NPML for evaluation
            if self.is_force_mle_eval:
                q_zCct = None
            if self.eval_use_crossentropy:
                loss = CrossEntropyLossLNPF.get_loss(self, p_yCc, z_samples, q_zCc, q_zCct, p_z, decoder_kl, Y_trgt)
            else:
                loss = NLLLossLNPF.get_loss(self, p_yCc, z_samples, q_zCc, q_zCct, p_z, decoder_kl, Y_trgt)

        if self.reduction is None:
            # size = [batch_size]
            return loss
        elif self.reduction == "mean":
            # size = [1]
            return loss.mean(0)
        elif self.reduction == "sum":
            # size = [1]
            return loss.sum(0)
        else:
            raise ValueError(f"Unknown {self.reduction}")

    @abc.abstractmethod
    def get_loss(self, p_yCc, z_samples, q_zCc, q_zCct, p_z, decoder_kl, Y_trgt):
        """Compute the Neural Process Loss

        Parameters
        ------
        p_yCc: torch.distributions.Distribution, batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
            Posterior distribution for target values {p(Y^t|y_c; x_c, x_t)}_t

        z_samples: torch.Tensor, size=[n_z_samples, batch_size, *n_lat, z_dim]
            Sampled latents. `None` if `encoded_path==deterministic`.

        q_zCc: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[z_dim]
            Latent distribution for the context points. `None` if `encoded_path==deterministic`.

        q_zCct: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[z_dim]
            Latent distribution for the targets. `None` if `encoded_path==deterministic`
            or not training or not `is_q_zCct`.

        Y_trgt: torch.Tensor, size=[batch_size, *n_trgt, y_dim]
            Set of all target values {y_t}.

        Return
        ------
        loss : torch.Tensor, size=[1].
        """
        pass


class CNPFLoss(BaseLossNPF):
    """Losss for conditional neural process (suf-)family [1]."""

    def get_loss(self, p_yCc, _, q_zCc, __, ___, ____, Y_trgt):
        assert q_zCc is None
        # \sum_t log p(y^t|z)
        # \sum_t log p(y^t|z). size = [z_samples, batch_size]
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)

        # size = [batch_size]
        nll = -sum_log_p_yCz.squeeze(0)
        return nll


class ELBOLossLNPF(BaseLossNPF):
    """Approximate conditional ELBO [1].

    References
    ----------
    [1] Garnelo, Marta, et al. "Neural processes." arXiv preprint
        arXiv:1807.01622 (2018).
    """

    def get_loss(self, p_yCc, _, q_zCc, q_zCct, __, ___, Y_trgt):

        # first term in loss is E_{q(z|y_cntxt,y_trgt)}[\sum_t log p(y^t|z)]
        # \sum_t log p(y^t|z). size = [z_samples, batch_size]
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)

        # E_{q(z|y_cntxt,y_trgt)}[...] . size = [batch_size]
        E_z_sum_log_p_yCz = sum_log_p_yCz.mean(0)

        # second term in loss is \sum_l KL[q(z^l|y_cntxt,y_trgt)||q(z^l|y_cntxt)]
        # KL[q(z^l|y_cntxt,y_trgt)||q(z^l|y_cntxt)]. size = [batch_size, *n_lat]
        kl_z = kl_divergence(q_zCct, q_zCc)
        # \sum_l ... . size = [batch_size]
        E_z_kl = sum_from_nth_dim(kl_z, 1)

        return -(E_z_sum_log_p_yCz - self.beta * E_z_kl)

class PACELBOLossLNPF(BaseLossNPF):
    """Approximate conditional ELBO [1].

    References
    ----------
    [1] Garnelo, Marta, et al. "Neural processes." arXiv preprint
        arXiv:1807.01622 (2018).
    """

    def get_loss(self, p_yCc, _, q_zCc, q_zCct, p_z, decoder_kl, Y_trgt):

        # first term in loss is E_{q(z|y_cntxt,y_trgt)}[\sum_t log p(y^t|z)]
        # \sum_t log p(y^t|z). size = [z_samples, batch_size]
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)

        # E_{q(z|y_cntxt,y_trgt)}[...] . size = [batch_size]
        E_z_sum_log_p_yCz = sum_log_p_yCz.mean(0)

        # second term in loss is \sum_l KL[q(z^l|y_cntxt,y_trgt)||q(z^l|y_cntxt)]
        # KL[q(z^l|y_cntxt,y_trgt)||q(z^l|y_cntxt)]. size = [batch_size, *n_lat]
        if q_zCct is not None:
            kl_z = kl_divergence(q_zCct, p_z)
        else:
            kl_z = kl_divergence(q_zCc, p_z)
        # \sum_l ... . size = [batch_size]
        E_z_kl = sum_from_nth_dim(kl_z, 1)

        decoder_kl = 0. if decoder_kl is None else decoder_kl

        return -(E_z_sum_log_p_yCz - self.beta * (E_z_kl + decoder_kl))

class PACMLossLNPF(BaseLossNPF):
    """PAC^m bound
    """

    def get_loss(self, p_yCc, _, q_zCc, q_zCct, p_z, decoder_kl, Y_trgt):
        n_z_samples, batch_size, *n_trgt = p_yCc.batch_shape

        # first term in loss is E_{q(z|y_cntxt,y_trgt)}[\sum_t log p(y^t|z)]
        # \sum_t log p(y^t|z). size = [z_samples, batch_size]
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)

        # E_{q(z|y_cntxt,y_trgt)}[...] . size = [batch_size]
        E_z_sum_log_p_yCz = torch.logsumexp(sum_log_p_yCz, dim=0) - math.log(n_z_samples)

        # # second term in loss is \sum_l KL[q(z^l|y_cntxt,y_trgt)||q(z^l|y_cntxt)]
        # # KL[q(z^l|y_cntxt,y_trgt)||q(z^l|y_cntxt)]. size = [batch_size, *n_lat]

        if q_zCct is not None:
            kl_z = kl_divergence(q_zCct, p_z)
        else:
            kl_z = kl_divergence(q_zCc, p_z)
        # \sum_l ... . size = [batch_size]
        E_z_kl = sum_from_nth_dim(kl_z, 1)

        decoder_kl = 0. if decoder_kl is None else decoder_kl

        return -(E_z_sum_log_p_yCz - self.beta * (E_z_kl + decoder_kl))

class PAC2LossLNPF(BaseLossNPF):
    """
    """

    def get_loss(self, p_yCc, _, q_zCc, q_zCct, p_z, decoder_kl, Y_trgt):
        # TODO: Make sure z_samples >= 2?
        # Make sure it's divisible by 2?

        # size = [n_z_samples, batch_size, *]
        log_p = p_yCc.log_prob(Y_trgt)

        log_p, log_p_ = torch.chunk(log_p, chunks=2, dim=0)

        eps = 0.1
        m_xi = (torch.maximum(log_p[0], log_p_[0]) + eps).detach()

        var_term = torch.exp(2*log_p - 2*m_xi) - torch.exp(log_p + log_p_ - 2*m_xi)
        sum_var = sum_from_nth_dim(var_term, 2)
        E_z_sum_var = sum_var.mean(0)

        # size = [n_z_samples, batch_size]
        sum_log_p = sum_from_nth_dim(log_p, 2)
        
        # first term in loss is E_{q(z|y_cntxt,y_trgt)}[\sum_t log p(y^t|z)]
        # \sum_t log p(y^t|z). size = [z_samples, batch_size]
        # sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)
        sum_log_p_yCz = sum_from_nth_dim(log_p, 2)

        # E_{q(z|y_cntxt,y_trgt)}[...] . size = [batch_size]
        E_z_sum_log_p_yCz = sum_log_p_yCz.mean(0)

        # second term in loss is \sum_l KL[q(z^l|y_cntxt,y_trgt)||q(z^l|y_cntxt)]
        # KL[q(z^l|y_cntxt,y_trgt)||q(z^l|y_cntxt)]. size = [batch_size, *n_lat]
        # kl_z = kl_divergence(q_zCct, q_zCc)

        if q_zCct is not None:
            kl_z = kl_divergence(q_zCct, p_z)
        else:
            kl_z = kl_divergence(q_zCc, p_z)
        
        # \sum_l ... . size = [batch_size]
        E_z_kl = sum_from_nth_dim(kl_z, 1)

        return -(E_z_sum_log_p_yCz + E_z_sum_var - self.beta * E_z_kl)

class PAC2TLossLNPF(BaseLossNPF):
    """
    """

    def get_loss(self, p_yCc, _, q_zCc, q_zCct, __, decoder_kl, Y_trgt):
        # TODO: Make sure z_samples >= 2?
        # Make sure it's divisible by 2?

        # size = [n_z_samples, batch_size, *]
        log_p = p_yCc.log_prob(Y_trgt)

        # size = [n_z_samples / 2, batch_size, *]
        log_p, log_p_ = torch.chunk(log_p, chunks=2, dim=0)

        eps = 0.1 # add ε for numerical stability
        # size = [batch_size, *]
        m_xi = (torch.maximum(log_p[0], log_p_[0]) + eps).detach()

        
        alpha_xi = (torch.logsumexp(
            torch.cat([torch.unsqueeze(log_p, dim=0), torch.unsqueeze(log_p_, dim=0)], dim=0), dim=0
        ) - m_xi - math.log(2)).detach()
        exp_alpha_xi = torch.exp(alpha_xi).detach()
        h_alpha_xi = (alpha_xi / torch.pow(1 - exp_alpha_xi, 2)) + \
            torch.pow(exp_alpha_xi * (1 - exp_alpha_xi), -1)
        h_alpha_xi.detach()
        
        
        var_term = h_alpha_xi * torch.exp(2*log_p - 2*m_xi) - torch.exp(log_p + log_p_ - 2*m_xi)
        sum_var = sum_from_nth_dim(var_term, 2)
        E_z_sum_var = sum_var.mean(0)

        # size = [n_z_samples, batch_size]
        sum_log_p = sum_from_nth_dim(log_p, 2)
        
        # first term in loss is E_{q(z|y_cntxt,y_trgt)}[\sum_t log p(y^t|z)]
        # \sum_t log p(y^t|z). size = [z_samples, batch_size]
        # sum_log_p_yCz, max_log_p_yCz = sum_log_prob_max(p_yCc, Y_trgt)
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)


        # E_{q(z|y_cntxt,y_trgt)}[...] . size = [batch_size]
        E_z_sum_log_p_yCz = sum_log_p_yCz.mean(0)

        # second term in loss is \sum_l KL[q(z^l|y_cntxt,y_trgt)||q(z^l|y_cntxt)]
        # KL[q(z^l|y_cntxt,y_trgt)||q(z^l|y_cntxt)]. size = [batch_size, *n_lat]
        # kl_z = kl_divergence(q_zCct, q_zCc)

        if q_zCct is not None:
            kl_z = kl_divergence(q_zCct, p_z)
        else:
            kl_z = kl_divergence(q_zCc, p_z)

        # \sum_l ... . size = [batch_size]
        E_z_kl = sum_from_nth_dim(kl_z, 1)

        return -(E_z_sum_log_p_yCz + E_z_sum_var - E_z_kl)


class NLLLossLNPF(BaseLossNPF):
    """
    Compute the approximate negative log likelihood for Neural Process family[?].

     Notes
    -----
    - might be high variance
    - biased
    - approximate because expectation over q(z|cntxt) instead of p(z|cntxt)
    - if q_zCct is not None then uses importance sampling (i.e. assumes that sampled from it).

    References
    ----------
    [?]
    """

    def get_loss(self, p_yCc, z_samples, q_zCc, q_zCct, _, __, Y_trgt):

        n_z_samples, batch_size, *n_trgt = p_yCc.batch_shape

        # computes approximate LL in a numerically stable way
        # LL = E_{q(z|y_cntxt)}[ \prod_t p(y^t|z)]
        # LL MC = log ( mean_z ( \prod_t p(y^t|z)) )
        # = log [ sum_z ( \prod_t p(y^t|z)) ] - log(n_z_samples)
        # = log [ sum_z ( exp \sum_t log p(y^t|z)) ] - log(n_z_samples)
        # = log_sum_exp_z ( \sum_t log p(y^t|z)) - log(n_z_samples)

        # \sum_t log p(y^t|z). size = [n_z_samples, batch_size]
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)

        # uses importance sampling weights if necessary
        if q_zCct is not None:

            # All latents are treated as independent. size = [n_z_samples, batch_size]
            sum_log_q_zCc = sum_log_prob(q_zCc, z_samples)
            sum_log_q_zCct = sum_log_prob(q_zCct, z_samples)

            # importance sampling : multiply \prod_t p(y^t|z)) by q(z|y_cntxt) / q(z|y_cntxt, y_trgt)
            # i.e. add log q(z|y_cntxt) - log q(z|y_cntxt, y_trgt)
            sum_log_w_k = sum_log_p_yCz + sum_log_q_zCc - sum_log_q_zCct
        else:
            sum_log_w_k = sum_log_p_yCz

        # log_sum_exp_z ... . size = [batch_size]
        log_S_z_sum_p_yCz = torch.logsumexp(sum_log_w_k, 0)

        # - log(n_z_samples)
        log_E_z_sum_p_yCz = log_S_z_sum_p_yCz - math.log(n_z_samples)

        # NEGATIVE log likelihood
        return -log_E_z_sum_p_yCz

class CrossEntropyLossLNPF(BaseLossNPF):
    """
    Compute the cross entropy loss from Masegosa
    """

    def get_loss(self, p_yCc, z_samples, q_zCc, q_zCct, _, __, Y_trgt):

        n_z_samples, batch_size, *n_trgt = p_yCc.batch_shape
        
        # size = [n_z_samples, batch_size, *]
        log_p = p_yCc.log_prob(Y_trgt)

        # size = [batch_size, *]
        logsumexp_p = torch.logsumexp(log_p, 0)

        log_S = sum_from_nth_dim(logsumexp_p - math.log(n_z_samples), 1) 

        return -log_S


#! might need gradient clipping as in their paper
class SUMOLossLNPF(BaseLossNPF):
    """
    Estimate negative log likelihood for Neural Process family using SUMO [1].

    Notes
    -----
    - approximate because expectation over q(z|cntxt) instead of p(z|cntxt)
    - if q_zCct is not None then uses importance sampling (i.e. assumes that sampled from it).

    Parameters
    ----------
    p_n_z_samples : scipy.stats.rv_frozen, optional
        Distribution for the number of of z_samples to take.

    References
    ----------
    [1] Luo, Yucen, et al. "SUMO: Unbiased Estimation of Log Marginal Probability for Latent
    Variable Models." arXiv preprint arXiv:2004.00353 (2020)
    """

    def __init__(
        self,
        p_n_z_samples=LightTailPareto(a=5).freeze(85),
        **kwargs,
    ):
        super().__init__()
        self.p_n_z_samples = p_n_z_samples

    def get_loss(self, p_yCc, z_samples, q_zCc, q_zCct, _, __, Y_trgt):

        n_z_samples, batch_size, *n_trgt = p_yCc.batch_shape

        # \sum_t log p(y^t|z). size = [n_z_samples, batch_size]
        sum_log_p_yCz = sum_log_prob(p_yCc, Y_trgt)

        # uses importance sampling weights if necessary
        if q_zCct is not None:
            # All latents are treated as independent. size = [n_z_samples, batch_size]
            sum_log_q_zCc = sum_log_prob(q_zCc, z_samples)
            sum_log_q_zCct = sum_log_prob(q_zCct, z_samples)

            #! It should be p(y^t,z|cntxt) but we are using q(z|cntxt) instead of p(z|cntxt)
            # \sum_t log (q(y^t,z|cntxt) / q(z|cntxt,trgt)) . size = [n_z_samples, batch_size]
            sum_log_w_k = sum_log_p_yCz + sum_log_q_zCc - sum_log_q_zCct
        else:
            sum_log_w_k = sum_log_p_yCz

        # size = [n_z_samples, 1]
        ks = (torch.arange(n_z_samples) + 1).unsqueeze(-1)
        #! slow to always put on GPU
        log_ks = ks.float().log().to(sum_log_w_k.device)

        #! the algorithm in the paper is not correct on ks[:k+1] and forgot inv_weights[m:]
        # size = [n_z_samples, batch_size]
        cum_iwae = logcumsumexp(sum_log_w_k, 0) - log_ks

        #! slow to always put on GPU
        # you want reverse_cdf which is P(K >= k ) = 1 - P(K < k) = 1 - P(K <= k-1) = 1 - CDF(k-1)
        inv_weights = torch.from_numpy(1 - self.p_n_z_samples.cdf(ks - 1)).to(
            sum_log_w_k.device
        )

        m = self.p_n_z_samples.support()[0]
        # size = [batch_size]
        sumo = cum_iwae[m - 1] + (
            inv_weights[m:] * (cum_iwae[m:] - cum_iwae[m - 1 : -1])
        ).sum(0)

        nll = -sumo
        return nll

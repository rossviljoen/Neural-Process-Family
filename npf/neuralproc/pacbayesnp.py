#!/usr/bin/env python3

"""Module for a PAC-Bayesian variant of the latent neural process"""
import torch
from npf.architectures import MergeFlatInputs

from .base import LatentNeuralProcessFamily
from .np import CNP

__all__ = ["PACBayesLNP"]

class PACBayesLNP(LatentNeuralProcessFamily, CNP):
    """
    PAC-Bayes Latent Neural Process
    """

    def __init__(self, x_dim, y_dim, **kwargs):
        kwargs = {
            k:kwargs[k] for k in kwargs
            if k not in {"encoded_path", "z_dim", "x_transf_dim"}
        }
        super().__init__(
            x_dim,
            y_dim,
            encoded_path="latent", # Ensure only the latent path is used
            z_dim=None,            # Ensure z_dim == r_dim
            x_transf_dim=-1,       # Ensure x_trans_dim == r_dim
            **kwargs
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.p_z = self.PriorDistribution(
            torch.zeros(self.z_dim, requires_grad=False, device=self.device),
            torch.ones(self.z_dim, requires_grad=False, device=self.device)
        )

    def forward(self, X_cntxt, Y_cntxt, X_trgt, Y_trgt=None):
        """
        Given a set of context feature-values {(x^c, y^c)}_c and target features {x^t}_t, return
        a set of posterior distribution for target values {p(Y^t|y_c; x_c, x_t)}_t.

        Parameters
        ----------
        X_cntxt: torch.Tensor, size=[batch_size, *n_cntxt, x_dim]
            Set of all context features {x_i}. Values need to be in interval [-1,1]^d.

        Y_cntxt: torch.Tensor, size=[batch_size, *n_cntxt, y_dim]
            Set of all context values {y_i}.

        X_trgt: torch.Tensor, size=[batch_size, *n_trgt, x_dim]
            Set of all target features {x_t}. Values need to be in interval [-1,1]^d.

        Y_trgt: torch.Tensor, size=[batch_size, *n_trgt, y_dim], optional
            Set of all target values {y_t}. Only required during training and if
            using latent path.

        Return
        ------
        p_y_trgt: torch.distributions.Distribution, batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
            Posterior distribution for target values {p(Y^t|y_c; x_c, x_t)}_t

        z_samples: torch.Tensor, size=[n_z_samples, batch_size, *n_lat, r_dim]
            Sampled latents. `None` if `encoded_path==deterministic`.

        q_zCc: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[r_dim]
            Latent distribution for the context points. `None` if `encoded_path==deterministic`.

        q_zCct: torch.distributions.Distribution, batch shape=[batch_size, *n_lat] ; event shape=[r_dim]
            Latent distribution for the targets. `None` if `encoded_path==deterministic`
            or not training or not `is_q_zCct`.
        """
        self.n_z_samples = (
            self.n_z_samples_train if self.training else self.n_z_samples_test
        )

        
        self._validate_inputs(X_cntxt, Y_cntxt, X_trgt, Y_trgt)

        # size = [batch_size, *n_cntxt, x_transf_dim]
        # with torch.no_grad():
            # Stop gradient in the encoding of X_cntxt for PAC-Bayes
        X_cntxt = self.x_encoder(X_cntxt) 
        # size = [batch_size, *n_trgt, x_transf_dim]
        X_trgt = self.x_encoder(X_trgt)

        # {R^u}_u
        # size = [batch_size, *n_rep, r_dim]
        R = self.encode_globally(X_cntxt, Y_cntxt)

        # encoded_path must be "latent"
        z_samples, q_zCc, q_zCct, p_z = self.latent_path(X_cntxt, R, X_trgt, Y_trgt)

        # size = [n_z_samples, batch_size, *n_trgt, r_dim]
        R_trgt = self.trgt_dependent_representation(X_cntxt, z_samples, None, X_trgt)

        # p(y|cntxt,trgt)
        # batch shape=[n_z_samples, batch_size, *n_trgt] ; event shape=[y_dim]
        p_yCc = self.decode(X_trgt, R_trgt)

        if isinstance(self.decoder, MergeFlatInputs):
            decoder_kl = self.decoder.flat_module.kl_q_p()
        else:
            decoder_kl = self.decoder.kl_q_p()

        try:
            if isinstance(self.x_encoder, MergeFlatInputs):
                decoder_kl += 0.01* self.x_encoder.flat_module.kl_q_p()
            else:
                decoder_kl += 0.01*self.x_encoder.kl_q_p()
        except AttributeError:
            pass

        return p_yCc, z_samples, q_zCc, q_zCct, p_z, decoder_kl
    
    def encode_globally(self, X_cntxt, Y_cntxt):
        X_cntxt = X_cntxt[0]    # Only use the first sample for the context encoding
        batch_size, n_cntxt, _ = X_cntxt.shape

        # encode all cntxt pair separately
        # size = [batch_size, n_cntxt, r_dim]
        R_cntxt = self.xy_encoder(X_cntxt, Y_cntxt)

        # using mean for aggregation (i.e. n_rep=1)
        # size = [batch_size, 1, r_dim]
        R = torch.mean(R_cntxt, dim=1, keepdim=True)

        if n_cntxt == 0:
            # arbitrarily setting the global representation to zero when no context
            R = torch.zeros(batch_size, 1, self.r_dim, device=R_cntxt.device)

        return R

    def trgt_dependent_representation(self, _, z_samples, __, X_trgt):

        n_z_samples, batch_size, n_trgt, _ = X_trgt.shape

        # size = [n_z_samples, batch_size, 1, z_dim]
        R_trgt = z_samples

        R_trgt = R_trgt.expand(n_z_samples, batch_size, n_trgt, self.r_dim)

        return R_trgt

    def latent_path(self, X_cntxt, R, X_trgt, Y_trgt):

        # q(z|c)
        # batch shape = [batch_size, *n_lat] ; event shape = [z_dim]
        q_zCc = self.infer_latent_dist(X_cntxt, R)

        if self.is_q_zCct and Y_trgt is not None:
            # during training when we know Y_trgt, we can take an expectation over q(z|cntxt,trgt)
            # instead of q(z|cntxt). note that actually does q(z|trgt) because trgt has cntxt
            R_from_trgt = self.encode_globally(X_trgt, Y_trgt)
            q_zCct = self.infer_latent_dist(X_trgt, R_from_trgt)
            sampling_dist = q_zCct
        else:
            q_zCct = None
            sampling_dist = q_zCc

        # size = [n_z_samples, batch_size, *n_lat, z_dim]
        z_samples = sampling_dist.rsample([self.n_z_samples])

        p_z = self.p_z.expand(q_zCc.batch_shape)

        return z_samples, q_zCc, q_zCct, p_z

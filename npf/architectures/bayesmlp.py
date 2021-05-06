#!/usr/bin/env python3

# A simple implementation of a Variational Bayesian MLP

# %%
import torch
import torch.nn as nn
import math

# %%
class BayesianMLP(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_size=32,
            n_hidden_layers=1,
            input_sampled=True,
            n_samples_train=1,
            n_samples_test=1,
            activation=nn.ReLU()
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers

        self.input_sampled = input_sampled
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test
        
        self.activation = activation

        # Need to add a sample dimension if x does not include one (only on the first layer though)
        self.to_hidden = BayesianLinear(
            self.input_size,
            self.hidden_size,
            input_sampled=self.input_sampled,
        )
        self.linears = nn.ModuleList(
            [
                BayesianLinear(self.hidden_size, self.hidden_size)
                for _ in range(self.n_hidden_layers - 1)
            ]
        )
        self.out = BayesianLinear(self.hidden_size, self.output_size)
        self.num_kl_params = (
            self.to_hidden.num_kl_params +
            sum(l.num_kl_params for l in self.linears) +
            self.out.num_kl_params
        )
        
    def forward(self, x):
        # x.shape = [n_z_samples, batch_size, input_size] if input_sampled=True
        # x.shape = [batch_size, input_size] if input_sampled=False
        # print("x shape:", x.shape)
        num_samples = self.n_samples_train if self.training else self.n_samples_test
        out = self.to_hidden(x, num_samples=num_samples)
        out = self.activation(out)

        for linear in self.linears:
            out = linear(out)
            out = self.activation(out)
            
        # out.shape = [n_samples, batch_size, output_size]
        out = self.out(out)
        return out

    def kl_q_p(self, reduction="sum"):
        first = self.to_hidden.kl_q_p()
        middles = sum(l.kl_q_p() for l in self.linears)
        last = self.out.kl_q_p()
        n = self.num_kl_params if reduction == "mean" else 1.
        return (first + middles + last) / n

    
# %%
class BayesianLinear(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            input_sampled=True,
            prior_mean=0.,
            prior_std=0.05
    ):
        super(BayesianLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.input_sampled = input_sampled
        
        self.p_mu = prior_mean
        self.p_log_sigma = math.log(prior_std)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.q_W_mu = nn.Parameter(torch.zeros((output_size, input_size), device=self.device))
        self.q_W_log_sigma = nn.Parameter(torch.ones((output_size, input_size), device=self.device))

        self.q_b_mu = nn.Parameter(torch.zeros(output_size, device=self.device))
        self.q_b_log_sigma = nn.Parameter(torch.ones(output_size, device=self.device))

        num_W_params = len(self.q_W_mu.view(-1)) + len(self.q_W_log_sigma.view(-1))
        num_b_params = len(self.q_b_mu.view(-1)) + len(self.q_b_log_sigma.view(-1))
        self.num_kl_params = num_W_params + num_b_params
        
        self.reset_parameters()

    def forward(self, x, num_samples=1):
        if self.input_sampled:
            # x.shape = [n_z_samples, batch_size, *, in_size]
            n_samples = x.shape[0]
            batch_size = x.shape[1]
        else:
            # x.shape = [batch_size, *, in_size]
            n_samples = num_samples
            batch_size = x.shape[0]
            # x.shape = [n_samples, batch_size, *, in_size]
            x = x.expand(n_samples, *x.shape)
        
        # W.shape = [n_samples, out_size, in_size]
        W = self.q_W_mu + torch.exp(self.q_W_log_sigma) * torch.randn((n_samples, *self.q_W_log_sigma.shape), device=self.device)
        b = self.q_b_mu + torch.exp(self.q_b_log_sigma) * torch.randn((n_samples, *self.q_b_log_sigma.shape), device=self.device)

        # out.shape = [n_samples, batch_size, *, out_size]
        out = torch.matmul(x.transpose(0, 1), W.transpose(1, 2)).transpose(0, 1)
        out += b.unsqueeze(1).unsqueeze(1)
        return out

    def reset_parameters(self):
        # Initialization method of Adv-BNN
        stdv = 1. / math.sqrt(self.q_W_mu.size(1))
        self.q_W_mu.data.uniform_(-stdv, stdv)
        self.q_W_log_sigma.data.fill_(self.p_log_sigma)
        self.q_b_mu.data.uniform_(-stdv, stdv)
        self.q_b_log_sigma.data.fill_(self.p_log_sigma)
        
    def kl_q_p(self):
        kl = self._kl(self.q_W_mu, self.q_W_log_sigma, self.p_mu, self.p_log_sigma)
        kl += self._kl(self.q_b_mu, self.q_b_log_sigma, self.p_mu, self.p_log_sigma)
        return kl

    def _kl(self, mu_1, log_sigma_1, mu_2, log_sigma_2):
        """
        mu_1, log_sigma_1 are tensors
        mu_2, log_sigma_2 are scalars
        """
        kl = (log_sigma_2 - log_sigma_1 +
              (torch.exp(log_sigma_1)**2 + (mu_1 + mu_2)**2) / (2 * math.exp(log_sigma_2)**2)) - 0.5
        return kl.sum()
        

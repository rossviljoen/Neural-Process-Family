#!/usr/bin/env python3

# A simple implementation of a Variational Bayesian MLP

# %%
import torch
import torch.nn as nn
import torch.distributions as td

# %%
class BayesianMLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=32,
        n_hidden_layers=1,
        activation=nn.ReLU()
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers

        self.activation = activation

        self.to_hidden = BayesianLinear(self.input_size, self.hidden_size)
        self.linears = nn.ModuleList(
            [
                BayesianLinear(self.hidden_size, self.hidden_size)
                for _ in range(self.n_hidden_layers - 1)
            ]
        )
        self.out = BayesianLinear(self.hidden_size, self.output_size)
        
        
    def forward(self, x):
        # x.shape = [n_z_samples, batch_size, input_size]
        # print("x shape:", x.shape)
        out = self.to_hidden(x)
        out = self.activation(out)

        for linear in self.linears:
            out = linear(out)
            out = self.activation(out)
            
        # out.shape = [n_samples, batch_size, output_size]
        out = self.out(out)
        return out

    def kl_q_p(self):
        first = self.to_hidden.kl_q_p()
        middles = sum(l.kl_q_p() for l in self.linears)
        last = self.out.kl_q_p()
        return first + middles + last

    
# %%
class BayesianLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(BayesianLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.p_W = td.Normal(
            torch.zeros((output_size, input_size), requires_grad=False, device=device),
            torch.ones((output_size, input_size), requires_grad=False, device=device)
        ) # weight prior
        self.p_b = td.Normal(
            torch.zeros(output_size, requires_grad=False, device=device),
            torch.ones(output_size, requires_grad=False, device=device)
        ) # bias prior

        self.q_W_mu = nn.Parameter(torch.zeros((output_size, input_size), device=device))
        self.q_W_log_sigma = nn.Parameter(torch.ones((output_size, input_size), device=device))
        # self.q_W_mu = nn.Parameter(torch.Tensor((output_size, input_size), device=device))
        # self.q_W_log_sigma = nn.Parameter(torch.Tensor((output_size, input_size), device=device))
        self.q_W = td.Normal(
            self.q_W_mu,
            torch.exp(self.q_W_log_sigma)
        ) # weight approximate posterior
        self.q_b_mu = nn.Parameter(torch.zeros(output_size, device=device))
        self.q_b_log_sigma = nn.Parameter(torch.ones(output_size, device=device))
        self.q_b = td.Normal(
            self.q_b_mu,
            torch.exp(self.q_b_log_sigma)
        ) # bias approxtimate posterior

    def forward(self, x):
        # x.shape = [n_z_samples, batch_size, *, in_size]
        n_samples = x.shape[0]
        batch_size = x.shape[1]
        # W.shape = [n_samples, out_size,xb in_size]
        self.q_W = td.Normal(
            self.q_W_mu,
            torch.exp(self.q_W_log_sigma)
        ) # weight approximate posterior
        W = self.q_W.rsample([n_samples])
        # b.shape = [n_samples, out_size]
        self.q_b = td.Normal(
            self.q_b_mu,
            torch.exp(self.q_b_log_sigma)
        ) # bias approxtimate posterior
        b = self.q_b.rsample([n_samples])

        # out.shape = [n_samples, batch_size, *, out_size]
        out = torch.matmul(x.transpose(0, 1), W.transpose(1, 2)).transpose(0, 1)
        out += b.unsqueeze(1).unsqueeze(1)
        return out

    def reset_parameters(self):
        # Initialization method of Adv-BNN
        stdv = 1. / math.sqrt(self.q_W_mu.size(1))
        self.q_W_mu.data.uniform_(-stdv, stdv)
        self.q_W_log_sigma.data.fill_(1.)
        self.q_b_mu.data.uniform_(-stdv, stdv)
        self.q_b_log_sigma.data.fill_(1.)
    
    def kl_q_p(self):
        return td.kl_divergence(self.q_W, self.p_W) + td.kl_divergence(self.q_b, self.p_b)

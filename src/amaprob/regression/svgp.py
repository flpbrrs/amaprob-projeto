"""SVGP (Processo Gaussiano Variacional Esparso) com média linear, ajustado por minibatch na ELBO.

Referências: Hensman, Fusi e Lawrence, "Gaussian Processes for Big Data" (2013); Rasmussen e
Williams, *Gaussian Processes for Machine Learning* (2006), cap. 2 e 5.

Modelo, sobre as colunas padronizadas φ(z) e o alvo padronizado t = (y − ȳ) / s_y:

    f ~ GP(m, k),   m(φ) = aᵀφ + b,   k(φ, φ') = σ_f² exp(−½ Σ_j (φ_j − φ'_j)² / ℓ_j²)
    t | f ~ N(f(φ), σ²)

A média linear faz do GP uma "regressão linear + correção não linear": longe dos dados, a
previsão volta para a reta, e não para zero. O kernel RBF tem um comprimento ℓ_j por coluna (ARD),
e a padronização deixa os comprimentos comparáveis entre as colunas.

Aproximação esparsa. M pontos indutores Z_u, com u = f(Z_u) e distribuição variacional
q(u) = N(m_u, S_u) (S_u com Cholesky completo). As posições Z_u também são aprendidas. Com
K_uu = k(Z_u, Z_u) e k_* = k(Z_u, φ_*), a marginal variacional de f num ponto é

    q(f_*) = N( m(φ_*) + k_*ᵀ K_uu⁻¹ (m_u − m(Z_u)),
                k_** − k_*ᵀ K_uu⁻¹ (K_uu − S_u) K_uu⁻¹ k_* )

Na implementação, u é parametrizado em coordenadas brancas (u = L v, L Lᵀ = K_uu), o que deixa a
otimização mais estável e não muda a marginal acima.

ELBO (limite inferior da evidência), que dispensa a matriz N × N:

    L = Σ_n E_{q(f_n)}[log N(t_n | f_n, σ²)] − KL(q(u) ‖ p(u))

A esperança tem forma fechada: log N(t_n | μ_n, σ²) − v_n / (2σ²), com μ_n e v_n os momentos de
q(f_n). Num minibatch B, a soma é estimada por (N / |B|) Σ_{n∈B}, sem viés. O custo por passo é
O(|B| M² + M³), e não O(N³).

Preditiva: y_* ~ N(μ_*, v_* + σ²), de volta na escala de y (média · s_y + ȳ, variância · s_y²).
O termo v_* cresce longe dos pontos indutores até σ_f², e é ele que alarga o intervalo em
refeições diferentes das do treino.

Colunas constantes no ajuste ficam com φ = 0 (desvio trocado por 1), como na regressão linear.
Toda a conta é em float64, para que o Cholesky de K_uu e as variâncias pequenas fiquem estáveis
sem aumentar o jitter.
"""

import time

import gpytorch
import numpy as np
import torch

TAMANHO_BLOCO_PREVISAO = 20_000


class _GPVariacional(gpytorch.models.ApproximateGP):
    def __init__(self, pontos_indutores: torch.Tensor):
        distribuicao = gpytorch.variational.CholeskyVariationalDistribution(pontos_indutores.shape[0])
        estrategia = gpytorch.variational.VariationalStrategy(
            self, pontos_indutores, distribuicao, learn_inducing_locations=True,
        )
        super().__init__(estrategia)
        d = pontos_indutores.shape[1]
        self.mean_module = gpytorch.means.LinearMean(input_size=d)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=d))

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class SVGP:
    def __init__(self, n_indutores: int = 256, epocas: int = 60, tamanho_lote: int = 1024,
                 taxa_aprendizado: float = 0.01, semente: int = 0):
        self.n_indutores = n_indutores
        self.epocas = epocas
        self.tamanho_lote = tamanho_lote
        self.taxa_aprendizado = taxa_aprendizado
        self.semente = semente

    def _phi(self, Z: np.ndarray) -> torch.Tensor:
        return torch.from_numpy((np.asarray(Z, dtype=np.float64) - self.media_z_) / self.desvio_z_)

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "SVGP":
        inicio = time.perf_counter()
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.media_z_ = Z.mean(axis=0)
        desvio = Z.std(axis=0)
        self.desvio_z_ = np.where(desvio > 0, desvio, 1.0)
        self.media_y_, self.desvio_y_ = y.mean(), y.std()
        Phi = self._phi(Z)
        t = torch.from_numpy((y - self.media_y_) / self.desvio_y_)
        N = len(t)

        torch.manual_seed(self.semente)
        linhas = np.random.default_rng(self.semente).choice(N, size=min(self.n_indutores, N), replace=False)
        self.modelo_ = _GPVariacional(Phi[linhas].clone()).double()
        self.verossimilhanca_ = gpytorch.likelihoods.GaussianLikelihood().double()
        self.modelo_.train()
        self.verossimilhanca_.train()
        otimizador = torch.optim.Adam(
            [*self.modelo_.parameters(), *self.verossimilhanca_.parameters()], lr=self.taxa_aprendizado,
        )
        elbo = gpytorch.mlls.VariationalELBO(self.verossimilhanca_, self.modelo_, num_data=N)
        embaralhador = torch.Generator().manual_seed(self.semente)

        self.historico_elbo_ = []
        for _ in range(self.epocas):
            soma, lotes = 0.0, 0
            for lote in torch.randperm(N, generator=embaralhador).split(self.tamanho_lote):
                otimizador.zero_grad()
                perda = -elbo(self.modelo_(Phi[lote]), t[lote])  # ELBO por refeição
                perda.backward()
                otimizador.step()
                soma, lotes = soma - perda.item(), lotes + 1
            self.historico_elbo_.append(soma / lotes)

        self.modelo_.eval()
        self.verossimilhanca_.eval()
        self.segundos_ajuste_ = time.perf_counter() - inicio
        return self

    @property
    def variancia_ruido_(self) -> float:
        return self.verossimilhanca_.noise.item() * self.desvio_y_**2

    def predict(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Phi = self._phi(Z)
        media, variancia = np.empty(len(Phi)), np.empty(len(Phi))
        with torch.no_grad():
            for inicio in range(0, len(Phi), TAMANHO_BLOCO_PREVISAO):
                b = slice(inicio, inicio + TAMANHO_BLOCO_PREVISAO)
                preditiva = self.verossimilhanca_(self.modelo_(Phi[b]))
                media[b] = preditiva.mean.numpy()
                variancia[b] = preditiva.variance.numpy()
        return media * self.desvio_y_ + self.media_y_, variancia * self.desvio_y_**2

"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
import json
import numpy as np
import torch as th

from .basic_ops import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type        # EPSILON
        self.model_var_type = model_var_type          # LEARNED_RANGE
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps    # default True

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)     # 输入参数，表示扩散模型的噪声调度参数，betas = np.linspace(0.0001, 0.02, 400)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])      # num_timesteps=400DDIM, 1000DDPM,基于beta的形状，扩散模型总时间步

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        model,
        x,
        t,
        y0=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        gamma=0.,
        num_update=1,
        regularizer=None,
        cond_kwargs=None,
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param y0: low-quality image
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param gamma: hyper-parameter to regularize the predicted x_start
                    0.5 * || x - x_0 ||_2 + gamma * R(x_0, y_0)      (1)
        :param num_update: number of update for x_start based on Eq. (1)
        :param regularizer: the constraint for R in Eq. (1)
        :param cond_kwargs: extra params for the regrlarizer
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:       # predict x_{t-1}
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            if gamma > 0:
                pred_xstart = self._refine_xstart(
                        pred_xstart,
                        y0=y0,
                        gamma=gamma,
                        num_update=num_update,
                        regularizer=regularizer,
                        cond_kwargs=cond_kwargs,
                        )
                pred_xstart = process_xstart(pred_xstart)
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
                pred_xstart = process_xstart(model_output)
            else:
                pred_noise = model_output
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )                                                  #  predict \eps
            if gamma > 0:
                pred_xstart = self._refine_xstart(
                        pred_xstart,
                        y0=y0,
                        gamma=gamma,
                        num_update=num_update,
                        regularizer=regularizer,
                        cond_kwargs=cond_kwargs,
                        )
                pred_xstart = process_xstart(pred_xstart)
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,          #模型预测的均值
            "variance": model_variance,  #模型预测的方差（可学习或者固定）
            "log_variance": model_log_variance,  #对数方差
            "pred_xstart": pred_xstart,  #预测的x0
            "pred_noise": pred_noise,    #预测的噪声
        }

    def _refine_xstart(self, x_start, y0, gamma, num_update=1, regularizer=None, cond_kwargs=None):
        """
        :param x_start: predicted x_0 by the diffusion model
        :param y0: low-quality image, [-1,1]
        :param gamma: hyper-parameter to regularize the predicted x_start
                    0.5 * || x - x_0 ||_2 + gamma * R(x_0, y_0)      (1)
        :param num_update: number of update for x_start based on Eq. (1)
        :param regularizer: the constraint for R in Eq. (1)
        :param cond_kwargs: extra params for the regrlarizer
        """
        if gamma <= 0 or regularizer is None:
            return x_start  # Skip refinement if gamma is 0 or regularizer is None

        if not callable(regularizer):
            raise ValueError("regularizer must be a callable function when gamma > 0")
        
        original_dtype = x_start.dtype
        x0_current = th.empty_like(x_start, dtype=th.float32).copy_(x_start)
        x0_current.requires_grad_(True)
        for _ in range(num_update):
            if cond_kwargs is None:
                loss = regularizer(y0, x0_current)
            else:
                loss = regularizer(y0, x0_current, cond_kwargs)
            grad = th.autograd.grad(
                        outputs=loss,
                        inputs=x0_current,
                        grad_outputs=None,
                                    )[0]
            assert grad.shape == x0_current.shape
            x0_update = x0_current.detach() - gamma * grad
            x0_current.data = x0_update.data

        return x0_update.type(original_dtype)

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self, model, x, t, y0,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        gamma=0.,
        num_update=1,
        regularizer=None,
        cond_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param y0: low-quality image
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param gamma: hyper-parameter to regularize the predicted x_start
                    0.5 * || x - x_0 ||_2 + gamma * R(x_0, y_0)      (1)
        :param num_update: number of update for x_start based on Eq. (1)
        :param regularizer: the constraint for R in Eq. (1)
        :param cond_kwargs: extra params for the regrlarizer
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            y0,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            gamma=gamma,
            num_update=num_update,
            regularizer=regularizer,
            cond_kwargs=cond_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        y0=None,
        noise=None,
        start_timesteps=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        gamma=0.,
        num_update=1,
        regularizer=None,
        cond_kwargs=None,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param y0: low-quality image
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param gamma: hyper-parameter to regularize the predicted x_start
                    0.5 * || x - x_0 ||_2 + gamma * R(x_0, y_0)      (1)
        :param num_update: number of update for x_start based on Eq. (1)
        :param regularizer: the constraint for R in Eq. (1)
        :param cond_kwargs: extra params for the regrlarizer
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            y0=y0,
            noise=noise,
            start_timesteps=start_timesteps,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            gamma=gamma,
            num_update=num_update,
            regularizer=regularizer,
            cond_kwargs=cond_kwargs,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        y0,
        noise=None,
        start_timesteps=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        gamma=0.,
        num_update=1,
        regularizer=None,
        cond_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        if start_timesteps is None:
            indices = list(range(self.num_timesteps))[::-1]
        else:
            assert noise is not None
            indices = list(range(start_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            out = self.p_sample(
                model,
                img,
                t,
                y0=y0,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                gamma=gamma,
                num_update=num_update,
                regularizer=regularizer,
                cond_kwargs=cond_kwargs,
            )
            yield out
            img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )       # q(x_{t-1}|x_t, x_0)
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [ ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE, ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)   # q(x_t|x_0)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)  # q(x_t|x_0)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }

    def ddim_sample(
        self,
        model,
        x,
        t,      #一个时刻的采样
        y0=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        gamma=0.0,
        num_update=1,
        regularizer=None,
        cond_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            y0,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            gamma=gamma,
            num_update=num_update,
            regularizer=regularizer,
            cond_kwargs=cond_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        if "pred_noise" in out:
            eps = out["pred_noise"]
        else:
            eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        y0=None,
        noise=None,
        start_timesteps=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        gamma=0.0,
        num_update=1,
        regularizer=None,
        cond_kwargs=None,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            y0=y0,
            noise=noise,
            start_timesteps=start_timesteps,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            gamma=gamma,
            num_update=num_update,
            regularizer=regularizer,
            cond_kwargs=cond_kwargs,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        y0=None,
        noise=None,
        start_timesteps=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        gamma=0.0,
        num_update=1,
        regularizer=None,
        cond_kwargs=None,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        if start_timesteps is None:
            indices = list(range(self.num_timesteps))[::-1]  #切片操作反转列表，逆序完整时间步
        else:
            assert noise is not None
            indices = list(range(start_timesteps))[::-1]     #切片操作反转列表，逆序部分时间步

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            out = self.ddim_sample(
                model,
                img,
                t,
                y0=y0,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                eta=eta,
                gamma=gamma,
                num_update=num_update,
                regularizer=regularizer,
                cond_kwargs=cond_kwargs,
            )
            yield out
            img = out["sample"]
    
    def get_timesteps(self, start_timesteps=None):
        if start_timesteps is None:
            return list(range(self.num_timesteps))[::-1]
        return list(range(start_timesteps))[::-1]

    def get_steps(self, min_t, max_t, num_steps, device):
        step_indices = th.arange(num_steps, dtype=th.float, device=device)  #创建步数索引
        alpha_min = 1 - min_t**2  # Convert sigma to alpha
        alpha_max = 1 - max_t**2
        alpha_steps = alpha_max + step_indices / (num_steps - 1) * (alpha_min - alpha_max)   #在alpha_min和alpha_max之间进行线性插值，生成num_steps个alpha值
        return th.sqrt(1 - alpha_steps)  # Convert back to sigma
    
    def sigma_to_t(self, sigma, sample_timesteps=None):
        """
        Convert noise level sigma to a time step index based on sampling timesteps.

        :param sigma: Noise level (scalar or tensor).
        :param sample_timesteps: List of time step indices used for sampling (e.g., [99, 89, ..., 0] for 100 steps).
        :return: Corresponding time step index in sampling schedule.
        """
        if sample_timesteps is None:
            raise ValueError("sample_timesteps must be provided for mapping sigma to sampling steps.")
        
        # Compute alpha = 1 - sigma^2
        alpha = 1 - sigma**2
        # Get alphas_cumprod for sampling timesteps
        sample_alphas_cumprod = th.tensor([self.alphas_cumprod[t] for t in sample_timesteps])
        # Find closest alpha index
        closest_i = min(
            range(len(sample_alphas_cumprod)),
            key=lambda i: abs(sample_alphas_cumprod[i] - alpha)
        )
        # Return the corresponding sampling time step
        return sample_timesteps[closest_i]
    
    def ddim_sample_loop_with_restart(
        self,
        model,
        shape,
        y0=None,
        noise=None,
        start_timesteps=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        gamma=0.0,
        num_update=1,
        regularizer=None,
        cond_kwargs=None,
        restart_info="",
        restart_gamma=0.0,
        restart_noise_scale=0.1,
    ):
        """
        Generate samples from the model using DDIM with Restart Sampling.

        Same output as ddim_sample_loop: returns final sample tensor.

        :param restart_info: JSON string for restart configuration, e.g., '{"10": [5, 3, 0.1, 0.5]}'.
        :param restart_gamma: noise increase factor for restart iterations.
        :param restart_noise_scale: scale of noise perturbation during restart.
        """
        final = None
        for sample in self.ddim_sample_loop_progressive_with_restart(
            model,
            shape,
            y0=y0,
            noise=noise,
            start_timesteps=start_timesteps,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            gamma=gamma,
            num_update=num_update,
            regularizer=regularizer,
            cond_kwargs=cond_kwargs,
            restart_info=restart_info,
            restart_gamma=restart_gamma,
            restart_noise_scale=restart_noise_scale,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive_with_restart(
        self,
        model,
        shape,
        y0=None,
        noise=None,
        start_timesteps=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        gamma=0.0,
        num_update=1,
        regularizer=None,
        cond_kwargs=None,
        restart_info="",
        restart_gamma=0.0,
        restart_noise_scale=0.1,
    ):
        """
        Generate samples with DDIM and Restart Sampling, yielding intermediate samples.

        Yields dictionaries with 'sample' and 'pred_xstart' keys, matching ddim_sample_loop_progressive.

        :param restart_info: JSON string for restart configuration, e.g., '{"10": [5, 3, 0.1, 0.5]}'.
        :param restart_gamma: noise increase factor for restart iterations.
        :param restart_noise_scale: scale of noise perturbation during restart.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = self.get_timesteps(start_timesteps)   #获取时间步列表，逆序[100,99,...,0]
        print('indices',indices)
        # Parse restart configuration
        # restart_info是JSON字符串
        # '{"10": [5, 3, 0.1, 0.5]}'键表示触发重启的时间步i+1，值[num_restart_steps, K, t_min, t_max]
        restart_list = json.loads(restart_info) if restart_info else {}
        #restart_list = {
        #    int(min(range(len(self.alphas_cumprod)), key=lambda i: abs(self.alphas_cumprod[i] - (1 - v[2]**2)))): v
        #    for k, v in restart_list.items()
        #}
        # 将用户提供的基于噪声级别（σ）的重启配置，映射到模型内部的时间步索引，同时进行正序到逆序时间步的转换。
        # 新增：存储原始时间步与映射后的时间步的对应关系
        #restart_mappings = []
        #restart_list = {
        #    int(len(self.alphas_cumprod) - 1 - 
        #        min(range(len(self.alphas_cumprod)), 
        #            key=lambda i: abs(self.alphas_cumprod[i] - (1 - v[2]**2)))): v
        #    for k, v in restart_list.items()
        #}
        restart_list = {int(k): v for k, v in json.loads(restart_info).items()} if restart_info else {}
        print('restart_list',restart_list)
        #解析后的restart_list是一个字典，结构为
        # {
        #       mapped_time_step_1: [num_restart_steps, K, t_min, t_max],单次重启的细化步数，重复重启的次数，重启噪声范围下限（σ值），重启噪声范围上限（σ值）

        #       mapped_time_step_2: [num_restart_steps, K, t_min, t_max],
        # }

        if progress:
            indices = tqdm(indices, desc="DDIM Sampling with Restart")

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            out = self.ddim_sample(
                model,
                img,
                t,
                y0=y0,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                eta=eta,
                gamma=gamma,
                num_update=num_update,
                regularizer=regularizer,
                cond_kwargs=cond_kwargs,
            )
            img = out["sample"]
            yield {"sample": img, "pred_xstart": out["pred_xstart"]}   #yield返回一个生成器

            # Restart operation
            if i in restart_list:   #如果此时到restart_list的时间步了
                num_restart_steps, K, t_min, t_max = restart_list[i]
                for _ in range(K):
                    new_t_steps = self.get_steps(t_min, t_max, num_restart_steps, device)
                    # Convert t_max to corresponding sampling time step
                    t_max_index = self.sigma_to_t(t_max, indices)
                    print('t_max_index',t_max_index)
                    # Forward diffusion using q_sample
                    img = self.q_sample(
                        x_start=img,
                        t=th.tensor([t_max_index] * shape[0], device=device),
                        noise=th.randn_like(img) * restart_noise_scale
                    )
                    #new_t_indices = [self.sigma_to_t(sigma.item()) for sigma in new_t_steps]
                    #new_t_indices = list(reversed(new_t_indices))  # 反转顺序
                    #print('new_t_indices',new_t_indices)
                    #img = img + th.randn_like(img) * restart_noise_scale * (new_t_steps[0]**2 - new_t_steps[-1]**2).sqrt()    #噪声扰动
                    #print(len(new_t_steps))
                    #for j in range(len(new_t_steps)):   #对应于前向加噪之后的反向DDIM采样10次
                    for t_sigma in new_t_steps:
                        #t_new = th.tensor([new_t_indices[j]] * shape[0], device=device) #时间步生成
                        #t_new = th.tensor([i+num_restart_steps-j-1] * shape[0], device=device)
                        t_new = th.tensor([self.sigma_to_t(t_sigma.item(), indices)] * shape[0], device=device)
                        print('t_new',t_new)
                        out = self.ddim_sample(
                            model,
                            img,
                            t_new,
                            y0=y0,
                            clip_denoised=clip_denoised,
                            denoised_fn=denoised_fn,
                            model_kwargs=model_kwargs,
                            eta=eta,
                            gamma=restart_gamma,
                            num_update=num_update,
                            regularizer=regularizer,
                            cond_kwargs=cond_kwargs,
                        )
                        img = out["sample"]
                    yield {"sample": img, "pred_xstart": out["pred_xstart"]}
    
    def ddim_sample_loop_progressive_restart(
        self,
        model,
        shape,
        y0=None,
        noise=None,
        start_timesteps=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        gamma=0.0,
        num_update=1,
        regularizer=None,
        cond_kwargs=None,
        k_restart=None,  # 新增：重启触发时间步
        K=None,          # 新增：重启过程总时间步
        k_skip=0,        # 新增：重启过程跳过的步数
    ):
        """
        DDIM 逐步采样，加入 DiffECC 重启机制以提升采样质量。

        Args:
            k_restart (int): 触发重启的时间步。
            K (int): 重启过程的总时间步数。
            k_skip (int): 重启过程中跳过的早期时间步数。
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        
        # 初始化图像
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        
        # 设置时间步
        if start_timesteps is None:
            indices = list(range(self.num_timesteps))[::-1]
        else:
            assert noise is not None
            indices = list(range(start_timesteps))[::-1]

        if progress:
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            out = self.ddim_sample(
                model,
                img,
                t,
                y0=y0,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                eta=eta,
                gamma=gamma,
                num_update=num_update,
                regularizer=regularizer,
                cond_kwargs=cond_kwargs,
            )

            # 重启机制 (步骤 12-23)
            if k_restart is not None and i == k_restart:
                # 保存当前预测 (步骤 14)
                x_r_0 = out["pred_xstart"]  # ˆx_r^0将当前时间步保存为x_r_0

                # 重新分配时间步 (步骤 13)，为重启过程生成新的时间步序列
                # 创建新的噪声调度表（此处简化为使用相同的调度，实际可重新设计）
                restart_indices = list(range(K))[::-1][k_skip:]  # 从 K-1 到 k_skip-1
                k_max = K - k_skip  # 最大时间步 (步骤 15)

                # 初始化重启状态 (步骤 16)
                t_k_max = th.tensor([k_max] * shape[0], device=device)
                x_k = self.q_sample(x_r_0, t_k_max, noise=th.randn(*shape, device=device))

                # 重启过程的逆向扩散 (步骤 17-21)
                for k in restart_indices:
                    t_k = th.tensor([k] * shape[0], device=device)
                    restart_out = self.ddim_sample(
                        model,
                        x_k,
                        t_k,
                        y0=y0,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        eta=eta,
                        gamma=gamma,
                        num_update=num_update,
                        regularizer=regularizer,
                        cond_kwargs=cond_kwargs,
                    )
                    x_k = restart_out["sample"]
                    if k == 0:
                        x_r_0 = restart_out["pred_xstart"]  # 更新最终预测

                # 更新主循环状态 (步骤 22，已使用 q_sample 替换)
                t_prev = th.tensor([i - 1] * shape[0], device=device)
                img = self.q_sample(x_r_0, t_prev, noise=th.randn(*shape, device=device))
            else:
                img = out["sample"]
            
            # 保存最终结果
            if i == 0:
                final_sample = out["pred_xstart"]  # t=0 时的 pred_x0 为最终结果

        return final_sample
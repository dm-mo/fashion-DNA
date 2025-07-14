import warnings
from typing import Optional, Any, Tuple

import gpytorch
import linear_operator
import torch

__all__ = ['MinibatchedDirichletClassificationLikelihood', 'MultitaskDirichletClassificationLikelihood']


# noinspection PyProtectedMember
class FixedNoiseMultitaskGaussianLikelihood(
    gpytorch.likelihoods.multitask_gaussian_likelihood._MultitaskGaussianLikelihoodBase
):
    has_global_noise = False

    def __init__(
            self,
            num_tasks: int,
            noise: torch.Tensor,
            learn_additional_noise: Optional[bool] = False,
            **kwargs
    ):
        super().__init__(
            num_tasks, gpytorch.likelihoods.noise_models.FixedGaussianNoise(noise=noise),
            0, None, torch.Size()
        )
        self.second_noise_covar: Optional[gpytorch.likelihoods.noise_models.HomoskedasticNoise] = None
        if learn_additional_noise:
            noise_prior = kwargs.get("noise_prior", None)
            noise_constraint = kwargs.get("noise_constraint", None)
            self.second_noise_covar = gpytorch.likelihoods.noise_models.HomoskedasticNoise(
                noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=torch.Size([num_tasks])
            )

    @property
    def noise(self) -> torch.Tensor:
        return self.noise_covar.noise + self.second_noise

    @property
    def second_noise(self) -> torch.Tensor:
        return torch.transpose(self.second_noise_covar.noise, -1, -2)

    def _shaped_noise_covar(
            self, shape: torch.Size, add_noise: Optional[bool] = True, interleaved: bool = True, *params: Any,
            noise_idx: Optional[torch.Tensor] = None, **kwargs: Any
    ):
        assert interleaved
        assert noise_idx is not None
        assert shape[:-1] == noise_idx.shape
        assert shape[-1] == self.noise.shape[-1]
        return gpytorch.linear_operator.operators.DiagLinearOperator(self.noise[noise_idx, :].view(-1))

    # Still needed? Better stub it, in case some methods calls this function
    def _eval_corr_matrix(self):
        raise NotImplementedError()


class MultitaskDirichletClassificationLikelihood(FixedNoiseMultitaskGaussianLikelihood):
    def __init__(
            self,
            targets: torch.Tensor,
            alpha_epsilon: float = 0.01,
            learn_additional_noise: Optional[bool] = False,
            dtype: torch.dtype = torch.float,
            **kwargs: Any,
    ):
        sigma2_labels, transformed_targets, num_classes = self._prepare_targets(
            targets, alpha_epsilon=alpha_epsilon, dtype=dtype
        )
        super().__init__(
            num_classes,
            noise=sigma2_labels,
            learn_additional_noise=learn_additional_noise,
            **kwargs,
        )
        self.transformed_targets: torch.Tensor = transformed_targets
        self.num_classes: int = num_classes
        self.targets: torch.Tensor = targets
        self.alpha_epsilon: float = alpha_epsilon

    @staticmethod
    def _prepare_targets(
            targets: torch.Tensor, alpha_epsilon: float = 0.01, dtype: torch.dtype = torch.float
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        num_classes = int(targets.max() + 1)

        # alpha[~class_labels] = alpha_epsilon
        alpha = alpha_epsilon * torch.ones(targets.shape[-1], num_classes, device=targets.device, dtype=dtype)
        # alpha[class_labels] = 1 + alpha_epsilon
        alpha[torch.arange(len(targets)), targets] += 1.0

        # sigma^2 = log(1 / alpha + 1)
        sigma2_labels = torch.log(alpha.reciprocal() + 1.0)

        # y = log(alpha) - 0.5 * sigma^2
        transformed_targets = alpha.log() - 0.5 * sigma2_labels
        return sigma2_labels.type(dtype), transformed_targets.type(dtype), num_classes


class MinibatchedDirichletClassificationLikelihood(gpytorch.likelihoods.DirichletClassificationLikelihood):
    def _shaped_noise_covar(self, base_shape, *params, noise_idx: Optional[torch.Tensor] = None, **kwargs):
        if len(params) > 0:
            # we can infer the shape from the params
            shape = None
        else:
            # here shape[:-1] is the batch shape requested, and shape[-1] is `n`, the number of points
            shape = base_shape
        assert noise_idx is not None
        assert len(noise_idx.shape) == 1
        assert shape[-1] == noise_idx.shape[0]
        res = self.noise_covar(*params, shape=shape, noise=self.noise_covar.noise[..., noise_idx], **kwargs)

        if self.second_noise_covar is not None:
            res = res + self.second_noise_covar(*params, shape=shape, **kwargs)
        elif isinstance(res, linear_operator.operators.ZeroLinearOperator):
            warnings.warn(
                "You have passed data through a FixedNoiseGaussianLikelihood that did not match the size "
                "of the fixed noise, *and* you did not specify noise. This is treated as a no-op.",
                gpytorch.utils.warnings.GPInputWarning,
            )

        return res

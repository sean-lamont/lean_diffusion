import os
import collections
import copy
import pickle

import fsspec
import numpy as np
import torch
import torch.nn.functional as F

import trainer_base
import utils
import algo


# - trainer_base -> diffusion -> Uniform state diffusion -> Duo base -> Duo -> Conditional Duo
# _loss -> nll (implemented by duo)

class CONDITIONAL_DUO(DUO):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)

        # todo properly index with condition cutoff


    def _loss(self, x0, valid_tokens,
              condition_cutoff,
              current_accumulation_step=None,
              train_mode=False):

        # todo make valid tokens for post condition tokens only
        (input_tokens, output_tokens,
         valid_tokens) = self._process_model_input(
            x0, valid_tokens)

        loss = self.nll(input_tokens, output_tokens, condition_cutoff,
                        current_accumulation_step, train_mode)

        assert loss.ndim == 2
        if self.ignore_bos:
            loss[:, 1:] = loss[:, 1:]
            valid_tokens[:, 1:] = valid_tokens[:, 1:]

        nlls = (loss * valid_tokens).sum()
        num_tokens = valid_tokens.sum()
        token_nll = nlls / num_tokens

        return Loss(loss=token_nll,
                    nlls=nlls,
                    prior_loss=0.0,
                    num_tokens=num_tokens)

    def _process_model_output(self, model_output, xt, sigma):
        del xt, sigma
        return model_output.log_softmax(dim=-1)

    # returns model output for non-conditional tokens only
    def forward(self, xt, condition_cutoff, sigma):

        sigma = self._process_sigma(sigma)

        # todo give attention mask to backbone??
        with torch.cuda.amp.autocast(dtype=torch.float32):
            model_output = self.backbone(xt, sigma)
            model_output = model_output[:, condition_cutoff:, :]

        # todo split up output to remove condition indices
        return self._process_model_output(
            model_output=model_output, xt=xt, sigma=sigma)

    def super_nll(self, x0, output_tokens, condition_cutoff,
                  current_accumulation_step=None, train_mode=False):

        del output_tokens

        t = self._sample_t(x0.shape[0],
                           current_accumulation_step)

        assert t.shape[0] == x0.shape[0]

        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            # t \in {1/T, 2/T, ..., 1}
            t += (1 / self.T)

        dalpha_t, alpha_t = self.noise(t)
        alpha_t = alpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        # split condition tokens before noising, and combine again after
        xt = self.q_xt(x0[:, condition_cutoff:], alpha_t)
        xt = torch.cat([x0[:, :condition_cutoff], xt], dim=1)

        log_x_theta = self.forward(xt, condition_cutoff, sigma=sigma)

        utils.print_nans(log_x_theta, 'model_output')

        return self.nll_per_token(
            log_x_theta=log_x_theta,
            xt=xt[:, condition_cutoff:],
            x0=x0[:, condition_cutoff:],
            alpha_t=alpha_t,
            dalpha_t=dalpha_t,
            low_var=train_mode and self.loss_type == 'low_var')

    def nll(self, x0, output_tokens, condition_cutoff,
            current_accumulation_step=None, train_mode=False):

        use_true_nll = (self.global_step > self.curriculum_end
                        or not train_mode)

        if use_true_nll:
            return super_nll(x0, output_tokens, condition_cutoff,
                             current_accumulation_step)
        del output_tokens


        t = self._sample_t(x0.shape[0], current_accumulation_step)

        gamma_t = self.gamma_min + t * (self.gamma_max
                                        - self.gamma_min)

        gamma_t_prime = self.gamma_max - self.gamma_min

        usdm_alpha_t = self._gamma_to_alphat(gamma_t)

        T = 1000

        usdm_dalpha_t = gamma_t_prime * T * (
                self._gamma_to_alphat(gamma_t + 1 / T) - usdm_alpha_t)

        usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
        usdm_dalpha_t = usdm_dalpha_t.unsqueeze(-1)
        assert usdm_alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(usdm_alpha_t)

        # todo what to do here for conditional tokens?
        x0_one_hot = F.one_hot(x0, self.vocab_size)

        # split one-hot here, don't want any noise given to conditional tokens
        xt = self._q_xt_gaussian(x0_one_hot[:, condition_cutoff:], gamma_t)
        xt = xt * self._compute_gumbel_tau_inverse()
        xt_usdm = xt.argmax(-1)

        # add back conditional tokens to xt
        xt = torch.cat([x0_one_hot[:, :condition_cutoff], xt], dim=1)

        # expect log_x_theta to be for non-conditional tokens only
        log_x_theta = self.forward(xt, condition_cutoff, sigma=sigma)

        return self.nll_per_token(log_x_theta=log_x_theta,
                                  xt=xt_usdm,
                                  x0=x0[:, condition_cutoff:],
                                  alpha_t=usdm_alpha_t,
                                  dalpha_t=usdm_dalpha_t,
                                  low_var=False)

    def training_step(self, batch, batch_idx):
        current_accumulation_step = (
                batch_idx % self.trainer.accumulate_grad_batches)
        # todo break input_ids into condition and input
        losses = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            batch['condition_cutoff'],
                            current_accumulation_step,
                            train_mode=True)
        self.metrics.update_train(losses.nlls, losses.prior_loss,
                                  losses.num_tokens)
        self.log(name='trainer/loss',
                 value=losses.loss.item(),
                 on_step=True,
                 on_epoch=False,
                 sync_dist=True, prog_bar=True)
        return losses.loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        losses = self._loss(batch['input_ids'],
                            batch['attention_mask'],
                            batch['condition_cutoff'],
                            )
        self.metrics.update_valid(losses.nlls, losses.prior_loss,
                                  losses.num_tokens)
        return losses.loss

    def on_validation_epoch_end(self):
        for k, v in self.metrics.valid_nlls.items():
            self.log(name=k, value=v.compute(), on_step=False,
                     on_epoch=True, sync_dist=True, prog_bar=True)
        if ((self.config.eval.compute_perplexity_on_sanity
             or not self.trainer.sanity_checking)
                and self.config.eval.generate_samples):
            samples, text_samples = None, None
            for _ in range(
                    self.config.sampling.num_sample_batches):
                samples = self.generate_samples(
                    num_samples=self.config.loader.eval_batch_size)

                self.metrics.record_entropy(samples)
                # Decode the samples to be re-tokenized by eval model
                text_samples = self.tokenizer.batch_decode(samples)
                if self.config.eval.compute_generative_perplexity:
                    self.metrics.record_generative_perplexity(
                        text_samples, self.num_tokens, self.device)
            if text_samples is not None:
                if self.trainer.global_rank == 0 and hasattr(
                        self.trainer.logger, 'log_table'):
                    # Log the last generated samples
                    text_samples = text_samples[
                        : self.config.sampling.num_sample_log]
                    self.trainer.logger.log_table(
                        key=f'samples@global_step{self.global_step}',
                        columns=['Generated Samples'],
                        data=[[s] for s in text_samples])
                if self.config.eval.compute_generative_perplexity:
                    self.log('val/gen_ppl',
                             self.metrics.gen_ppl.compute(),
                             on_epoch=True,
                             on_step=False,
                             sync_dist=True)
                    self.log('val/sample_entropy',
                             self.metrics.sample_entropy.compute(),
                             on_epoch=True,
                             on_step=False,
                             sync_dist=True)
        self._train_mode()

    # todo need conditions here to generate samples for (store random subset of validation conditions in internal variable?)
    @torch.no_grad()
    def generate_samples(self, sample_conditions, num_steps=None,
                         eps=1e-5):
        """Generate samples from the model."""

        num_samples = sample_conditions.shape[0]

        # Lightning auto-casting is not working in this method for some reason
        if num_steps is None:
            num_steps = self.config.sampling.steps
        x = self.prior_sample(num_samples, self.num_tokens)
        timesteps = torch.linspace(
            1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        # todo give ancestral update the conditions
        for i in range(num_steps):
            t = timesteps[i] * torch.ones(
                x.shape[0], 1, device=self.device)
            if self.sampler == 'ancestral':
                _, x = self._ancestral_update(
                    x=x, t=t, dt=dt, p_x0=None)
            elif self.sampler == 'ancestral_cache':
                p_x0_cache, x_next = self._ancestral_update(
                    x=x, t=t, dt=dt, p_x0=p_x0_cache)
                if (not torch.allclose(x_next, x)
                        or self.time_conditioning):
                    # Disable caching
                    p_x0_cache = None
                x = x_next
            else:
                x = self._analytic_update(x=x, t=t, dt=dt)

        t0 = timesteps[-1] * torch.ones(x.shape[0], 1,
                                        device=self.device)
        if self.config.sampling.noise_removal == 'ancestral':
            # not implemented (or valid?) for uniform state
            if self.sampler == 'analytic':
                x = self._denoiser_update(x=x, t=t0)
            else:
                _, x = self._ancestral_update(x=x, t=t0, dt=None,
                                              p_x0=p_x0_cache,
                                              noise_removal_step=True)
        elif self.config.sampling.noise_removal == 'greedy':
            sigma = self._sigma_from_alphat(self.noise(t0)[1])
            x = self.forward(xt=x, sigma=sigma).argmax(dim=-1)
        return x

    def _ancestral_update(self, x, t, dt, p_x0=None,
                          noise_removal_step=False):
        del p_x0
        _, alpha_t = self.noise(t)
        if noise_removal_step:
            alpha_s = torch.ones_like(alpha_t)
        else:
            _, alpha_s = self.noise(t - dt)
        sigma_t = self._sigma_from_alphat(alpha_t)
        assert alpha_t.ndim == 2

        # todo x here is without conditions, add the conditions to the forward x with
        q_xs = self._compute_posterior(
            x=self.forward(x, sigma_t).exp(),
            xt=x,
            alpha_s=alpha_s,
            alpha_t=alpha_t)
        if self.p_nucleus < 1:
            q_xs = utils.top_k_top_p_filtering(
                q_xs.log(), top_p=self.p_nucleus)
        return None, trainer_base.sample_categorical(q_xs)

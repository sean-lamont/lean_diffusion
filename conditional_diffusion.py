import random

import torch
import torch.nn.functional as F

import trainer_base
from trainer_base import Loss
import utils
from algo import *

# - trainer_base -> diffusion -> Uniform state diffusion -> Duo base -> Duo -> Conditional Duo
# _loss -> nll (implemented by duo)

'''
Conditional DUO model. Inherits from DUO model.
Conditioning is done by replacing the noised tokens with the original tokens at the condition indices.  
Cannot simply use an attention mask, as we need to replace the noised tokens with the original tokens before they are seen by the model.

'''


class CONDITIONAL_DUO(DUO):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)

        self.tmp_val_batches = []

    def _loss(self, x0, valid_tokens,
              condition_cutoff,
              current_accumulation_step=None,
              train_mode=False):

        (input_tokens, output_tokens,
         valid_tokens) = self._process_model_input(
            x0, valid_tokens)


        # todo could just pass condition mask and valid tokens from dataloader?
        # update valid tokens to include 0's at condition indices
        batch_size, seq_len = x0.shape
        device = x0.device
        condition_mask = torch.arange(seq_len, device=device).unsqueeze(0) < condition_cutoff.unsqueeze(1)

        # original duo is assumed to treat invalid tokens (i.e. padding) by default, as no attention mask is given to the model?
        loss = self.nll(input_tokens, output_tokens, condition_mask, valid_tokens,
                        current_accumulation_step, train_mode)

        # negate condition mask, and join with valid (i.e. valid must be not padding and not condition)
        # valid_tokens = valid_tokens * (~condition_mask).long()

        # print (f'before: {valid_tokens}')

        valid_tokens = torch.where(condition_mask, torch.tensor(0, dtype=valid_tokens.dtype, device=valid_tokens.device), valid_tokens)

        # print (f'after: {valid_tokens}')
        # print (torch.sum(valid_tokens, dim=1))


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
    def forward(self, xt, sigma, attention_mask=None, ):

        sigma = self._process_sigma(sigma)

        # todo give attention mask to backbone?
        with torch.cuda.amp.autocast(dtype=torch.float32):
            model_output = self.backbone(input_ids=xt, sigma=sigma, attention_mask=attention_mask)

        return self._process_model_output(
            model_output=model_output, xt=xt, sigma=sigma)

    def super_nll(self, x0, output_tokens, condition_mask, attention_mask,
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

        xt = self.q_xt(x0, alpha_t)

        xt = torch.where(condition_mask, x0, xt)

        log_x_theta = self.forward(xt, sigma=sigma, attention_mask=attention_mask)

        utils.print_nans(log_x_theta, 'model_output')

        return self.nll_per_token(
            log_x_theta=log_x_theta,
            xt=xt,
            x0=x0,
            alpha_t=alpha_t,
            dalpha_t=dalpha_t,
            low_var=train_mode and self.loss_type == 'low_var')

    def nll(self, x0, output_tokens, condition_mask, attention_mask,
            current_accumulation_step=None, train_mode=False):

        use_true_nll = (self.global_step > self.curriculum_end
                        or not train_mode)

        if use_true_nll:
            return self.super_nll(x0, output_tokens, condition_mask, attention_mask,
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

        x0_one_hot = F.one_hot(x0, self.vocab_size)

        xt = self._q_xt_gaussian(x0_one_hot, gamma_t)
        xt = xt * self._compute_gumbel_tau_inverse() # t -> 0, approaches arg max
        xt_usdm = xt.argmax(-1)

        ## add back conditional tokens to xt

        xt_usdm = torch.where(condition_mask, x0, xt_usdm)

        # todo is this expensive? better way to do this?
        one_hot_condition_mask = condition_mask.unsqueeze(-1).expand(-1, -1, xt.shape[-1])


        xt = torch.where(one_hot_condition_mask, x0_one_hot, xt)

        # note model has to support forward with different shapes / modes for soft vs hard tokens
        # given input_embeds as this is the 'curriculum learning' version where we pass in soft tokens (for non-conditional)


        # from paper, assumes the model forward step will apply softmax to the soft tokens
        log_x_theta = self.forward(xt, sigma=sigma, attention_mask=attention_mask, )

        utils.print_nans(log_x_theta, 'model_output')

        # eqn 48 in paper, log_x_theta is x_theta, xt is zt, x0 is x
        return self.nll_per_token(log_x_theta=log_x_theta,
                                  xt=xt_usdm,
                                  x0=x0,
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
        # randomly choose batch to store
        if len(self.tmp_val_batches) < self.config.sampling.num_sample_batches:
            if random.random() < 0.1:
                self.tmp_val_batches.append(batch)


        # if batch_idx < self.config.sampling.num_sample_batches:
        #     self.tmp_val_batches.append(batch)

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
            try:
                for i in range(
                        self.config.sampling.num_sample_batches):
                    samples = self.generate_samples(self.tmp_val_batches[i])

                    self.metrics.record_entropy(samples)
                    # Decode the samples to be re-tokenized by eval model

                    text_samples = self.tokenizer.batch_decode(samples)
                    if self.config.eval.compute_generative_perplexity:
                        self.metrics.record_generative_perplexity(
                            text_samples, self.num_tokens, self.device)
                    ground_truth = self.tokenizer.batch_decode(
                        self.tmp_val_batches[i]['input_ids'])
                if text_samples is not None:
                    if self.trainer.global_rank == 0 and hasattr(
                            self.trainer.logger, 'log_table'):
                        # Log the last generated samples
                        text_samples = text_samples[
                            : self.config.sampling.num_sample_log]

                        self.trainer.logger.log_table(
                            key=f'samples@global_step{self.global_step}',
                            columns=['Generated Samples'],
                            data=[[s, ground_truth[i]] for i, s in enumerate(text_samples)])

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

            except Exception as e:
                print(f'Error generating samples: {e}')

            self.tmp_val_batches = []
        self._train_mode()

    @torch.no_grad()
    def generate_samples(self, sample_conditions, num_steps=None,
                         eps=1e-5):
        """Generate samples from the model."""


        # assume num_tokens >= max seq len. Then we can just generate noise for whole vector, and set the conditioned tokens where needed
        # given a batch with input_ids, condition_cutoff, attention_mask
        condition_mask = torch.arange(sample_conditions['input_ids'].shape[1], device=sample_conditions['input_ids'].device).unsqueeze(0) < sample_conditions['condition_cutoff'].unsqueeze(1)

        num_samples = sample_conditions['input_ids'].shape[0]

        # Lightning auto-casting is not working in this method for some reason
        if num_steps is None:
            num_steps = self.config.sampling.steps
        # uniform random over vocab
        x = self.prior_sample(num_samples, self.num_tokens)

        timesteps = torch.linspace(
            1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            # reset x for each step
            x = torch.where(condition_mask, sample_conditions['input_ids'], x)

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
            x = torch.where(condition_mask, sample_conditions['input_ids'], x)

            # not implemented (or valid?) for uniform state
            if self.sampler == 'analytic':
                x = self._denoiser_update(x=x, t=t0)
            else:
                _, x = self._ancestral_update(x=x, t=t0, dt=None,
                                              p_x0=p_x0_cache,
                                              noise_removal_step=True)
        elif self.config.sampling.noise_removal == 'greedy':
            x = torch.where(condition_mask, sample_conditions['input_ids'], x)

            sigma = self._sigma_from_alphat(self.noise(t0)[1])
            x = self.forward(xt=x, sigma=sigma).argmax(dim=-1)

        x = torch.where(condition_mask, sample_conditions['input_ids'], x)
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

        q_xs = self._compute_posterior(
            x=self.forward(x, sigma_t).exp(),
            xt=x,
            alpha_s=alpha_s,
            alpha_t=alpha_t)

        if self.p_nucleus < 1:
            q_xs = utils.top_k_top_p_filtering(
                q_xs.log(), top_p=self.p_nucleus)
        return None, trainer_base.sample_categorical(q_xs)

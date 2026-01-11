import torch
import tools

class joint_model(torch.nn.Module):
    def __init__(self, config, world_model, policy):
        super().__init__()
        self.config = config
        self.wm = world_model
        self.policy = policy
        self.device = config.device

    def _warmup_rssm(self, data):
        """
        With sequence of observations, reset and warm up the RSSM within the world model, 
        and return the rssm_state as an output.
        """
        data = self.wm.preprocess(data)
        embed = self.wm.encoder(data)
        post, _ = self.wm.dynamics.observe(embed, data['action'], data['is_first'])
        # Return the last state for continuation
        return {k: v[:, -1] for k, v in post.items()}

    def observe(self, data):
        """
        Take in a batch of observations, according to world model settings: 
        preprocess, standardize, and encode. Then, observe, and output the post, prior.
        """
        data = self.wm.preprocess(data)
        embed = self.wm.encoder(data)
        post, prior = self.wm.dynamics.observe(embed, data['action'], data['is_first'])
        return post, prior

    def feats(self, state):
        """
        Returns feat using the WM
        """
        return self.wm.dynamics.get_feat(state)

    def get_preds(self, feats):
        """
        Use world model heads to perform predictions on the feats.
        Returns image predictions as tensors (not distributions), with optional
        destandardization when dataset stats are available.
        """
        preds = {}
        for name, head in self.wm.heads.items():
            pred = head(feats)
            if isinstance(pred, dict):
                preds.update(pred)
            else:
                preds[name] = pred
        
        if "image" in preds:
            image_pred = preds["image"]
            if hasattr(image_pred, "mode"):
                image_pred = image_pred.mode()
            if self.config.image_standardize:
                if self.wm._dataset_image_mean is not None and self.wm._dataset_image_std is not None:
                    mean = self.wm._dataset_image_mean
                    std = self.wm._dataset_image_std
                    while mean.dim() < image_pred.dim():
                        mean = mean.unsqueeze(0)
                        std = std.unsqueeze(0)
                    image_pred = image_pred * std + mean
            preds["image"] = image_pred
        return preds

    def obs_2_act(self, obs, state=None, action=None, is_first=None):
        """
        Get actions, taking in observations, updating world model, and outputing action by feeding feats into policy.
        """
        obs = self.wm.preprocess(obs)
        embed = self.wm.encoder(obs)
        B = embed.shape[0]

        if action is None:
             action = torch.zeros((B, self.config.num_actions), device=self.device)
        if is_first is None:
             if 'is_first' in obs:
                 is_first = obs['is_first']
             elif state is None:
                 is_first = torch.ones(B, device=self.device)
             else:
                 is_first = torch.zeros(B, device=self.device)
             
        post, _ = self.wm.dynamics.obs_step(state, action, embed, is_first, sample=self.training)
        
        feat = self.wm.dynamics.get_feat(post)
        new_action = self.act(feat)
            
        return new_action, post

    def act(self, inputs):
        """
        Take inputs, feed into policy, get action back. 
        Inputs are not nessaserily feats, to use this function it is up to the user to know what the policy should be fed.
        """
        action =  self.policy(inputs)
        if getattr(self.config, 'clip_actions', False):
            action = torch.clamp(action, -1.0, 1.0)
        return action

    def imagine(self, start_state, horizon):
        """
        Imagine a rollout from the current world model state and policy. 
        Take horizon as a length and compute what following the policy inside the world model would look like over horizon steps. 
        Returns sequence of features, sequence of states, and sequence of actions.
        """
        dynamics = self.wm.dynamics
        
        # Handle start_state shape (B, T, D) -> (B, D) if needed
        first_key = next(iter(start_state.keys()))
        if start_state[first_key].ndim == 3:
             start = {k: v[:, -1] for k, v in start_state.items()}
        else:
             start = start_state

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            
            action = self.act(inp)

            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon, device=self.device)], (start, None, None)
        )
        return feats, succ, actions

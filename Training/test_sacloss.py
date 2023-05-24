# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:26:25 2023

@author: napat
"""

import torch
from Training.learning05_discretesac import SAC_Agent

unityenv = UnityEnvironment()

sac_agent = SAC_Agent(unityenv)
sac_agent._initialise_memory(epsilon=1)
sac_agent._initialise_memory()
sac_agent.train(10, report_freq=1);

samples = sac_agent.sample_memory(32)

batch_data = list(map(list, zip(*samples)))

batch_states = torch.tensor(np.array(batch_data[0]))
batch_actions = torch.tensor(np.array(batch_data[1])).long()#.type(torch.int64)
batch_rewards = torch.tensor(np.array(batch_data[2])).unsqueeze(1).repeat(1,7)
batch_nextstates = torch.tensor(np.array(batch_data[3]))
batch_done = torch.tensor(np.array(batch_data[4])).float().unsqueeze(1).repeat(1,7)

with torch.no_grad():
    prob, log_prob = sac_agent.get_action_prob(batch_nextstates)
    next_q1 = sac_agent.target_critic1(batch_nextstates)
    next_q2 = sac_agent.target_critic2(batch_nextstates)
    next_q = torch.min(next_q1, next_q2).size()
    soft_state = prob * (next_q - sac_agent.alpha * log_prob)
    soft_state = soft_state.sum(dim=-1)
    target_q = batch_rewards + (1-batch_done) * sac_agent.gamma * soft_state
    
pred_q1 = sac_agent.critic1(batch_states)
soft_q1 = pred_q1.gather(-1, batch_actions.unsqueeze(-1)).squeeze(-1)
pred_q2 = sac_agent.critic2(batch_states)
soft_q2 = pred_q2.gather(-1, batch_actions.unsqueeze(-1)).squeeze(-1)

loss1 = sac_agent.criterion(target_q, soft_q1)
loss2 = sac_agent.criterion(target_q, soft_q2)

# batch_rewards.unsqueeze(-1).expand(32,7)
# batch_rewards.unsqueeze(1).repeat(1,7)


prob, log_prob = sac_agent.get_action_prob(batch_states)
q_values_local = sac_agent.critic1(batch_states)
q_values_local2 = sac_agent.critic2(batch_states)
inside_term = sac_agent.alpha * log_prob - torch.min(q_values_local, q_values_local2)
policy_loss = (prob * inside_term).sum(dim=-1).mean()
return policy_loss, log_prob

# Choose ACtion
actions = []
state = s
if not isinstance(state, torch.Tensor):
    state = torch.tensor(state) 
with torch.no_grad():
    action_prob = sac_agent.actor(state)
for i in range(sac_agent.n_branches):
    prob = action_prob[i,...]
    prob = np.array(prob)
    prob = prob/(prob.sum())
    a = np.random.choice(np.arange(sac_agent.n_actions), p=prob)
    actions.append(a)
np.random.choice(torch.arange(3).repeat(7, 1), p=prob)
    
actions = torch.multinomial(action_prob, 1)
actions = np.array(actions)




# Method Version

batch_states = torch.tensor(np.array(batch_data[0]))
batch_actions = torch.tensor(np.array(batch_data[1])).long()#.type(torch.int64)
batch_rewards = torch.tensor(np.array(batch_data[2])).unsqueeze(1).repeat(1,7)
batch_nextstates = torch.tensor(np.array(batch_data[3]))
batch_done = torch.tensor(np.array(batch_data[4])).float().unsqueeze(1).repeat(1,7)


with torch.no_grad():
    prob, log_prob = self.get_action_prob(batch_nextstates)
    next_q1 = self.target_critic1(batch_nextstates)
    next_q2 = self.target_critic2(batch_nextstates)
    next_q = torch.min(next_q1, next_q2).size()
    soft_state = prob * (next_q - self.alpha * log_prob)
    soft_state = soft_state.sum(dim=-1)
    target_q = batch_rewards + (1-batch_done) * self.gamma * soft_state
    
pred_q1 = self.critic1(batch_states)
soft_q1 = pred_q1.gather(-1, batch_actions.unsqueeze(-1)).squeeze(-1)
pred_q2 = self.critic2(batch_states)
soft_q2 = pred_q2.gather(-1, batch_actions.unsqueeze(-1)).squeeze(-1)

loss1 = self.criterion(target_q, soft_q1)
loss2 = self.criterion(target_q, soft_q2)

return loss1, loss2